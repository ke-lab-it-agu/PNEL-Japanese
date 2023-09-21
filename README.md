# PNEL-Japanese
![PNEL](PNEL1.png)

## usage
基本的に[pnel](https://github.com/debayan/pnel)を参考にmodelを構築しています。
以下python3.7環境を想定しています。

### setup
pip install
```
(pnel-ja)$ git clone https://github.com/ke-lab-it-agu/PNEL-Japanese.git
(pnel-ja)$ pip install -r requirements.txt
```
create indices and mappings as specified in deploy/data/esdumps/mappings.json
```
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_description.json --output=http://localhost:9200/wikidataentitydescriptionsindex01 --type=data
(pnel-ja)$ elasticdump --limit=10000 --input=japanese_label.json --output=http://localhost:9200/wikidataentitylabelindex01 --type=data
(pnel-ja)$ cd deploy/data
(pnel-ja)$ wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz
(pnel-ja)$ gunzip wikidata_translation_v1.tsv.gz
(pnel-ja)$ python loadwikiembeds.py
```
### start the server
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
```
(pnel-ja)$ 
(pnel-ja)$ 
```
### training
```
(pnel-ja)$ 
(pnel-ja)$ 
```
### evaluation
```
(pnel-ja)$ python parse.py
(pnel-ja)$ python judge.py
```
### pre-trained model
