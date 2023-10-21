import zipfile
import pathlib
import shutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.generation import GenerationConfig

# Extracting model weights
cur_dir = str(pathlib.Path().resolve())
with zipfile.ZipFile(f'{cur_dir}/text-detoxification/models/best.zip', 'r') as zip_ref:
    zip_ref.extractall(f'{cur_dir}')

# loading the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('best')
model.eval()
model.config.use_cache = False
model_checkpoint = "t5-small"
prefix = 'detoxify:'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.max_new_tokens = 128


def translate(inference_request, model, tokenizer=tokenizer):
    """Inference function with decoding"""
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, generation_config=gen_cfg)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

# Top 19 toxic sentences from dataset 
examples = [
    'His father would have used a booming voice to wrench a reply from this stupid machine.',
    'You have to send those idiots back in.',
    'Salina could be with that stupid cop.',
    "And don't let those idiots in radiology hold you up.",
    'My idiot friend here brought marijuana... - on the plane.',
    'That stupid couple told her to break up with you.',
    'The idiot, Max. He never should have sold it to you guys.',
    "Fuck you, soft. You're panicking",
    "It's coz of those two idiots. They escaped through a tunnel.",
    'You have to let this stupid plane crash make the decision for you.',
    "Audrey Cruz is an idiot. He's the boss.",
    "Why don't you get a job. Instead of playing golf with your stupid buddies?",
    'How do you like your stupid game now, Tom?',
    'Think about that shit, dawg.',
    'You idiots! You have betrayed the revolution.',
    'Why is this idiot Silencer shouting so much?',
    'Just like the rest of the stupid Irish in this country.',
    'Your shit is so tired, Justice.',
    'More than the stupid senators and congressmen that passed it.'
]
paraphrases = []

for example in examples:
    paraphrases.append(translate(prefix + example, model, tokenizer))

for ex, par in zip(examples, paraphrases):
    print(f'Original:    {ex}\nParaphrase:  {par}')

shutil.rmtree('best')
