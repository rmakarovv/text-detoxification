import wget
import os

if not os.path.exists('text-detoxification/data/external/paradetox.tsv'):
    url = 'https://huggingface.co/datasets/s-nlp/paradetox/resolve/main/train.tsv'
    wget.download(url, out='text-detoxification/data/external/paradetox.tsv')

if not os.path.exists('text-detoxification/data/raw/filtered_paranmt.zip'):
    url = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
    wget.download(url, out='text-detoxification/data/raw/filtered_paranmt.zip')
