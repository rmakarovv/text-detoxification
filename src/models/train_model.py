import zipfile
import os
import shutil
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import locale
locale.getpreferredencoding = lambda: "UTF-8"


# Reading preprocessed data
cur_dir = str(pathlib.Path().resolve())
data = pd.read_csv(f'{cur_dir}/text-detoxification/data/interim/preprocessed.csv')

# Setting up training model and tokenizer parameters
transformers.set_seed(42)
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
prefix = "detoxify:"
max_input_length = 70
max_target_length = 70

def preprocess_function(df):
    """Preprocessing inputs to include prefix and tokenizing them"""
    inputs = [prefix + text for text in df['reference']]
    targets = [text for text in df['translation']]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train = data.sample(frac = 0.8)
val = data.drop(train.index)
train.head()

# Preprocessing inputs
CHOOSE = 300000 # debug value. Current value is bigger than the whole dataset to include all sentences
cropped_datasets = {}
cropped_datasets['train'] = train.iloc[:CHOOSE, :]
cropped_datasets['val'] = val.iloc[:CHOOSE // 4, :]
tokenized_datasets = {}
tokenized_datasets['train'] = preprocess_function(cropped_datasets['train'])
tokenized_datasets['val'] = preprocess_function(cropped_datasets['val'])

samples_train = []
for i in range(len(tokenized_datasets['train']['input_ids'])):
    samples_train.append({'input_ids': tokenized_datasets['train']['input_ids'][i],
                          'attention_mask': tokenized_datasets['train']['attention_mask'][i],
                          'labels': tokenized_datasets['train']['labels'][i]})

samples_val = []
for i in range(len(tokenized_datasets['val']['input_ids'])):
    samples_val.append({'input_ids': tokenized_datasets['val']['input_ids'][i],
                          'attention_mask': tokenized_datasets['val']['attention_mask'][i],
                          'labels': tokenized_datasets['val']['labels'][i]})

tokenized_datasets['train'] = samples_train
tokenized_datasets['val'] = samples_val


"""Fine-tuning the t5 small model"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# defining the parameters for training
batch_size = 32
num_epochs = 3

model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    report_to='tensorboard',
    disable_tqdm=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

from sentence_transformers import SentenceTransformer, util
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    embeddings1 = bert_model.encode([x for x in decoded_preds],  convert_to_tensor=True)
    embeddings2 = bert_model.encode([x for x in decoded_labels], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    result = {"bert_simil": np.mean(np.diagonal(cosine_scores.cpu().numpy())).round(3)}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# saving model
trainer.save_model('best')
model.config.to_json_file("best/config.json")

# Loading log history and saving the graph of training
logs = pd.DataFrame(trainer.state.log_history)

eval_logs = logs['eval_loss'].dropna().reset_index(drop=True)
simil_logs = logs['eval_bert_simil'].dropna().reset_index(drop=True)
eval_epochs = np.array([i+1 for i in range(len(eval_logs))])
simil_epochs = np.array([i+1 for i in range(len(simil_logs))])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.set_title('Similarity')
ax2.set_title('Eval loss')
ax1.plot(simil_epochs, simil_logs)
ax2.plot(eval_epochs, eval_logs)
try:
    fig.savefig("text-detoxification/reports/figures/training.pdf", bbox_inches='tight')
except Exception as e:
    print(str(e))

shutil.make_archive('text-detoxification/models/best', 'zip', '', 'best')
shutil.rmtree(f'{model_name}-finetuned')
shutil.rmtree('best')
