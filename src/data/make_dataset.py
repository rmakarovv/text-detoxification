import wget

url = 'https://huggingface.co/datasets/s-nlp/paradetox/resolve/main/train.tsv'

file = wget.download(url, out='../../data/external/paradetox.tsv')
