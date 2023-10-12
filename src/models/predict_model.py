from  transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import zipfile

with zipfile.ZipFile('../../models/best.zip', 'r') as zip_ref:
    zip_ref.extractall('')

# loading the model and run inference for it
model = AutoModelForSeq2SeqLM.from_pretrained('best')
model.eval()
model.config.use_cache = False
model_checkpoint = "t5-small"
prefix = 'detoxify:'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def translate(inference_request, model, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True,temperature=0)

sents = ['i am so fucking bad', "Don't look, come or think of comming back! Tosser.",
         "Tony Sidaway is obviously a fistfuckee. He loves an arm up his ass."]
translated_sents = []

for sent in sents:
    translated_sents.append(translate(prefix + sent.lower(), model, tokenizer))

for i in range(len(sents)):
    print(f'Original sentence: "{sents[i]}"\nTranslated sentence: "{translated_sents[i]}"')
