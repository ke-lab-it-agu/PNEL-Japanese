#!/usr/bin/python
# python3.6or7で実行する

from flask import request
from flask import Flask
from gevent.pywsgi import WSGIServer
from elasticsearch import Elasticsearch
import gensim
import numpy as np
# import spacy
# import itertools
import json,sys
import random

# nlp = spacy.load('ja_ginza_electra')
# postags = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","","X"]
# es = Elasticsearch(port=9200)
app = Flask(__name__)

# cache = {}
AllCount = 0
MissCount = 0

def ConvertVectorSetToVecAverageBased(vectorSet, ignore = []):
    if len(ignore) == 0:
        return np.mean(vectorSet, axis = 0)
    else:
        return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)


print("TextMatch initializing, loading fastext")
try:
    es = Elasticsearch()
    model = gensim.models.KeyedVectors.load_word2vec_format('chive-1.2-mc5.tar')
    print("loaded fastext")
except Exception as e:
    print(e)
    sys.exit(1)


@app.route('/ftwv_tok', methods=['POST'])
def ftwv_tok():
    d = request.get_json(silent=True)
    chunks = d['chunks'] 
    vectors = []
    for chunk in chunks:
        print(chunk)
        # if chunk in cache:
        #     print('in cache')
        #     vectors.append(cache[chunk])
        #     continue
        print(chunk)
        phrase_1 = chunk.split(" ")
        vw_phrase_1 = []
        # global AllCount
        # AllCount = AllCount + 1
        for phrase in phrase_1:
            try:
                # print phrase
                vw_phrase_1.append(model.word_vec(phrase))
            except:
                # print traceback.print_exc()
                continue
        if len(vw_phrase_1) == 0:
           vectors.append(300*[0.0])
        #    cache[chunk] = 300*[0.0]
        #    global MissCount
        #    MissCount = MissCount + 1
        else:
           x = ConvertVectorSetToVecAverageBased(vw_phrase_1).tolist()
           vectors.append(x)
        #    cache[chunk] = x
    # print("AllCount:" + str(AllCount))
    # print("MissCount:" + str(MissCount))
    return json.dumps(vectors)

@app.route('/ftwv_dec', methods=['POST'])
def ftwv_dec():
    d = request.get_json(silent=True)
    chunks = d['chunks'] 
    vectors = []
    for chunk in chunks:
        print(chunk)
        # if chunk in cache:
        #     print('in cache')
        #     vectors.append(cache[chunk])
        #     continue
        print(chunk)
        phrase_1 = chunk.split(" ")
        vw_phrase_1 = []
        for phrase in phrase_1:
            try:
                # print phrase
                vw_phrase_1.append(model.word_vec(phrase))
            except:
                # print traceback.print_exc()
                continue
        if len(vw_phrase_1) == 0:
           vectors.append(300*[0.0])
        #    cache[chunk] = 300*[0.0]1
        else:
           x = ConvertVectorSetToVecAverageBased(vw_phrase_1).tolist()
           vectors.append(x)
        #    cache[chunk] = x
    return json.dumps(vectors)

@app.route('/textMatch', methods=['POST'])
def textMatch():
    pagerankflag = False
    d = request.get_json(silent=True)
    if 'pagerankflag' in d:
        pagerankflag = d['pagerankflag']
    chunks = d['chunks']
    matchedChunks = []
    for chunk in chunks:
         if chunk['class'] == 'entity':
             res = es.search(index="wikidataentitylabelindex03", body={"query":{"multi_match":{"query":chunk['chunk'],"fields":["wikidataLabel"]}},"size":200})
             _topkents = []
             topkents = []
             for record in res['hits']['hits']:
                 _topkents.append([record['_source']['uri'],record['_source']['wikidataLabel']])#record['_source']['edgecount']))
             #if pagerankflag:
             #    _topkents =  sorted(_topkents, key=lambda k: k[1], reverse=True)
             for record in _topkents:
                 if len(topkents) >= 50:
                     break
                 if record[0] in topkents:
                     continue
                 else:
                     topkents.append([record[0][37:],record[1]])
             matchedChunks.append({'chunk':chunk, 'topkmatches': topkents, 'class': 'entity'})
                 
    return json.dumps(matchedChunks)


if __name__ == '__main__':
    http_server = WSGIServer(('', int(sys.argv[1])), app)
    http_server.serve_forever()
                          
                     


#if __name__ == '__main__':
#    t = TextMatch()
    #print t.textMatch([{'chunk': 'Who', 'surfacelength': 3, 'class': 'entity', 'surfacestart': 0}, {'chunk': 'the parent organisation', 'surfacelength': 23, 'class': 'relation', 'surfacestart': 7}, {'chunk': 'Barack Obama', 'surfacelength': 12, 'class': 'entity', 'surfacestart': 34}, {'chunk': 'is', 'surfacelength': 2, 'class': 'relation', 'surfacestart': 4}])
#    print t.textMatch([{"chunk": "India", "surfacelength": 5, "class": "entity", "surfacestart": 0}])
