from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import zipfile

with zipfile.ZipFile('text-detoxification/models/best.zip', 'r') as zip_ref:
    zip_ref.extractall('best')

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

while True:
    sent = input('Input sentence that you want to detoxify:')
    print('Detoxified text:', translate(prefix + sent.lower(), model, tokenizer))
