import zipfile
import pathlib
import shutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.generation import GenerationConfig


cur_dir = str(pathlib.Path().resolve())

with zipfile.ZipFile(f'{cur_dir}/text-detoxification/models/best.zip', 'r') as zip_ref:
    zip_ref.extractall('best')

# loading the model and run inference for it
model = AutoModelForSeq2SeqLM.from_pretrained('best')
model.eval()
model.config.use_cache = False
model_checkpoint = "t5-small"
prefix = 'detoxify:'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.max_new_tokens = 128

def translate(inference_request, model, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, generation_config=gen_cfg)
    return tokenizer.decode(outputs[0], skip_special_tokens=True,temperature=0)

while True:
    try:
        sent = input('Input sentence that you want to detoxify or press ENTER to exit:\n')
        if not sent:
            break
        print('Detoxified text:', translate(prefix + sent.lower(), model, tokenizer))
    except KeyboardInterrupt:
        break

shutil.rmtree('best')
