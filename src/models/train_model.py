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
max_input_length = 128
max_target_length = 128

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
CHOOSE = 130000
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
num_epochs = 5

model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    report_to='none',
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# saving model
trainer.save_model('best')
model.config.to_json_file("best/config.json")

logs = pd.DataFrame(trainer.state.log_history)

eval_logs = logs['eval_loss'].dropna().reset_index(drop=True)
train_logs = logs['train_loss'].dropna().reset_index(drop=True)
eval_epochs = np.array([i+1 for i in range(len(eval_logs))])
train_epochs = np.array([i+1 for i in range(len(train_logs))])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.set_title('Train loss')
ax2.set_title('Eval loss')
ax2.plot(eval_epochs, eval_logs)
ax1.plot(train_epochs, train_logs)

fig.savefig("text-detoxification/reports/figures/training.pdf", bbox_inches='tight')

shutil.make_archive('text-detoxification/models/best', 'zip', '', 'best')
shutil.rmtree(f'{model_name}-finetuned')
shutil.rmtree('best')
