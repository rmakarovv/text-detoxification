import wget
import os
import pathlib
import zipfile
import pandas as pd
import shutil


if not os.path.exists('text-detoxification/data/external/paradetox.tsv'):
    url = 'https://huggingface.co/datasets/s-nlp/paradetox/resolve/main/train.tsv'
    wget.download(url, out='text-detoxification/data/external/paradetox.tsv')

if not os.path.exists('text-detoxification/data/raw/filtered_paranmt.zip'):
    url = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
    wget.download(url, out='text-detoxification/data/raw/filtered_paranmt.zip')

cur_dir = str(pathlib.Path().resolve())

with zipfile.ZipFile(f'{cur_dir}/text-detoxification/data/raw/filtered_paranmt.zip', 'r') as zip_ref:
    zip_ref.extractall('filtered_paranmt')
    
filtered = pd.read_table('filtered_paranmt/filtered.tsv')
filtered = filtered.drop([filtered.columns[i] for i in [0]], axis=1)

data = pd.read_table(f'{cur_dir}/text-detoxification/data/external/paradetox.tsv')
data.rename(columns={"en_toxic_comment": "reference", "en_neutral_comment": 'translation'}, inplace=True)

data2 = filtered[(filtered['ref_tox'] > 0.95) & (filtered['trn_tox'] < 0.01)]
data2 = data2.drop([data2.columns[i] for i in [2, 3, 4, 5]], axis=1)

final_data = pd.concat([data, data2])
final_data = final_data.reset_index(drop=True)
final_data.to_csv(f'{cur_dir}/text-detoxification/data/interim/preprocessed.csv')

shutil.rmtree('filtered_paranmt')
