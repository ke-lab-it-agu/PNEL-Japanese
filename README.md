# PNEL-Japanese
![PNEL](PNEL1.png)

## usage
PNEL-Japanese is built on the basis of [pnel](https://github.com/debayan/pnel).

We assume python 3.7 environment.

### setup
pip install
```
(pnel-ja)$ git clone https://github.com/ke-lab-it-agu/PNEL-Japanese.git && cd PNEL-Japanese
(pnel-ja)$ mkdir envs && cd envs
(pnel-ja)$ python3.7 -m venv pnel-ja
(pnel-ja)$ source pnel-ja/bin/activate
(pnel-ja)$ cd ../ && pip install -r requirements.txt
```
Create indices and mappings as specified in deploy/data/esdumps/mappings.json.

Wikidata information can be downloaded from [here](https://drive.google.com/drive/folders/1jpsypPoQzXioaDgerK-E3fLlLHtshtXP?usp=drive_link).
```
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_description.json --output=http://localhost:9200/wikidataentitydescriptionsindex01 --type=data
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_label.json --output=http://localhost:9200/wikidataentitylabelindex01 --type=data
(pnel-ja)$ cd deploy/data
(pnel-ja)$ wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz
(pnel-ja)$ gunzip wikidata_translation_v1.tsv.gz
(pnel-ja)$ python loadwikiembeds.py
```

### start the server
Set up a local server.

It depends on the language model you want to use.

・fastText
```
(pnel-ja)$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
(pnel-ja)$ python TextMatchServer_fastText.py 8887
```
・Wikipedia2Vec
```
(pnel-ja)$ wget http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.txt.bz2
(pnel-ja)$ python TextMatchServer_Wikipedia2Vec.py 8887
```
・chive
```
(pnel-ja)$ wget https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/chive-1.2-mc5.tar.gz
(pnel-ja)$ python TextMatchServe_chiver.py 8887
```
・WikiEntVec
```
(pnel-ja)$ wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.all_vectors.300d.txt.bz2
(pnel-ja)$ python TextMatchServer_WikiEntVec.py 8887
```

### vectorize
Get the embeddings you need.
```
(pnel-ja)$ cd vectorise
(pnel-ja)$ python preparedatangramtextmatchdesc.py datasets/japanese_train_translate.json webqtrain webqtrainvectors.txt
(pnel-ja)$ python preparedatangramtextmatchdesc.py datasets/japanese_test_translate.json webqtest webqtestvectors.txt
(pnel-ja)$ mkdir webqtrainchunks
(pnel-ja)$ cd webqtrainchunks
(pnel-ja)$ split -l 10 ../webqtrainvectors.txt webqchunk
```

### training
```
(pnel-ja)$ cd train
(pnel-ja)$ CUDA_VISIBLE_DEVICES=0 python -u train.py --data_path ../vectorise/webqtrainchunks/ --test_data_path ../vectorise/webqtestchunks.txt --models_dir ./models/webqmodels/
```

### evaluation
```
(pnel-ja)$ cd eval/webqsp
(pnel-ja)$ python parse.py
(pnel-ja)$ python judge.py
```
