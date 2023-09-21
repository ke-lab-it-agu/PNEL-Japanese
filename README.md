# PNEL-Japanese
![PNEL](PNEL1.png)

## usage
基本的に[pnel](https://github.com/debayan/pnel)を参考にmodelを構築しています

以下python3.7環境を想定しています


### setup
pip install
```
(pnel-ja)$ git clone https://github.com/ke-lab-it-agu/PNEL-Japanese.git
(pnel-ja)$ pip install -r requirements.txt
```

create indices and mappings as specified in deploy/data/esdumps/mappings.json

wikidataの各種情報はxxxからダウンロードできます
```
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_description.json --output=http://localhost:9200/wikidataentitydescriptionsindex01 --type=data
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_label.json --output=http://localhost:9200/wikidataentitylabelindex01 --type=data
(pnel-ja)$ cd deploy/data
(pnel-ja)$ wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz
(pnel-ja)$ gunzip wikidata_translation_v1.tsv.gz
(pnel-ja)$ python loadwikiembeds.py
```

### start the server
使用したい言語モデルによってそれぞれ以下のようにローカルサーバーを立ち上げます

・fastText
```
(pnel-ja)$ wget xxx
(pnel-ja)$ python TextMatchServer_fastText.py 8887
```
・Wikipedia2Vec
```
(pnel-ja)$ wget xxx
(pnel-ja)$ python TextMatchServer_Wikipedia2Vec.py 8887
```
・chive
```
(pnel-ja)$ wget xxx
(pnel-ja)$ python TextMatchServe_chiver.py 8887
```
・WikiEntVec
```
(pnel-ja)$ wget xxx
(pnel-ja)$ python TextMatchServer_WikiEntVec.py 8887
```

### vectorize
モデルの訓練を行うにあたって、あらかじめ必要な埋め込みを獲得します
```
(pnel-ja)$ cd vectorise
(pnel-ja)$ python preparedatangramtextmatchdesc.py datasets/japanese_train_translate.json webqtrain webqtrainvectors.txt
(pnel-ja)$ python preparedatangramtextmatchdesc.py datasets/japanese_test_translate.json webqtest webqtestvectors.txt
(pnel-ja)$ mkdir webqtrainchunks
(pnel-ja)$ cd webqtrainchunks
(pnel-ja)$ split -l 10 ../webqtrainvectors.txt webqchunk
```

### training
獲得した埋め込みを使用してモデルを訓練します
```
(pnel-ja)$ cd train
(pnel-ja)$ CUDA_VISIBLE_DEVICES=0 python -u train.py --data_path ../vectorise/webqtrainchunks/ --test_data_path ../vectorise/webqtestchunks.txt --models_dir ./models/webqmodels/
```

### evaluation
構築したモデルの評価を行います
```
(pnel-ja)$ cd eval/webqsp
(pnel-ja)$ python parse.py
(pnel-ja)$ python judge.py
```
