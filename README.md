# PNEL-Japanese
![PNEL](PNEL1.png)

## usage
PNEL-Japanese is built on the basis of [pnel](https://github.com/debayan/pnel).

We assume python 3.7 environment.

### setup
pip install
```
$ git clone https://github.com/ke-lab-it-agu/PNEL-Japanese.git && cd PNEL-Japanese
$ mkdir envs && cd envs
$ python3.7 -m venv pnel-ja
$ source pnel-ja/bin/activate
(pnel-ja)$ cd ../ && pip install -r requirements.txt
```
Create indices and mappings as specified in deploy/data/esdumps/mappings.json.

Wikidata information can be downloaded from [here](https://drive.google.com/drive/folders/1jpsypPoQzXioaDgerK-E3fLlLHtshtXP?usp=drive_link).
```
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_description.json --output=http://localhost:9200/wikidataentitydescriptionsindex03 --type=data
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_label.json --output=http://localhost:9200/wikidataentitylabelindex03 --type=data
(pnel-ja)$ cd deploy/data
(pnel-ja)$ wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz
(pnel-ja)$ gunzip wikidata_translation_v1.tsv.gz
(pnel-ja)$ cd ../utils && python loadwikiembeds.py
```

### start the server
Set up a local server.

It depends on the language model you want to use.

・fastText
```
(pnel-ja)$ cd deploy && wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
(pnel-ja)$ gunzip cc.ja.300.vec.gz
(pnel-ja)$ python fastText_TextMatchServer.py 8887
```
・Wikipedia2Vec
```
(pnel-ja)$ cd deploy && wget http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.txt.bz2
(pnel-ja)$ bunzip2 jawiki_20180420_300d.txt.bz2
(pnel-ja)$ python Wikipedia2Vec_TextMatchServer.py 8887
```
・chive
```
(pnel-ja)$ cd deploy && wget https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/chive-1.2-mc5.tar.gz
(pnel-ja)$ gunzip chive-1.2-mc5.tar.gz
(pnel-ja)$ python chive_TextMatchServer.py 8887
```
・WikiEntVec
```
(pnel-ja)$ cd deploy && wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.all_vectors.300d.txt.bz2
(pnel-ja)$ bunzip2 jawiki.all_vectors.300d.txt.bz2
(pnel-ja)$ python WikiEntVec_TextMatchServer.py 8887
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
(pnel-ja)$ CUDA_VISIBLE_DEVICES=0 python -u train.py --data_path ../vectorise/webqtrainchunks/ --test_data_path ../vectorise/webqtestvectors.txt --models_dir ./models/webqmodels/
```

### evaluation
```
(pnel-ja)$ python api.py --port 4444 --modeldir ../train/models/webqmodels/ --layers 1 --rnnsize 512 --attentionsize 128
(pnel-ja)$ cd eval/webqsp
(pnel-ja)$ python parse.py
(pnel-ja)$ python judge.py
```
