#!/usr/bin/python

import argparse
from flask import request, Response
from flask import Flask
from gevent.pywsgi import WSGIServer
import json,sys,requests,logging
import json
from Vectoriser import Vectoriser
from PointerNetworkLinker import PointerNetworkLinker
from pynif import NIFCollection
from rdflib import URIRef

logging.basicConfig(filename='earl.log',level=logging.INFO)

parser = argparse.ArgumentParser(description='Run PNEL API')
parser.add_argument('--port', dest='port')
parser.add_argument('--modeldir', dest='modeldir')
parser.add_argument('--rnnsize', dest='rnnsize')
parser.add_argument('--attentionsize', dest='attentionsize')
parser.add_argument('--layers', dest='layers')
args = parser.parse_args()

v = Vectoriser()
p = PointerNetworkLinker(args.modeldir,int(args.rnnsize),int(args.attentionsize),int(args.layers))
#j = JointLinker()

app = Flask(__name__)

@app.route('/processQuery', methods=['POST'])
def processQuery():
    d = request.get_json(silent=True)
    print(d)
    nlquery = d['nlquery']
    print(len(nlquery.split()), nlquery)
    if len(nlquery.split()) > 20:
        return json.dumps({'nlquery':d['nlquery'],'entities':[], 'error': 'Sentence can not contain more than 20 words'}, indent=4, sort_keys=True)
    pagerankflag = False
    try:
        if 'pagerankflag' in d:
            pagerankflag = d['pagerankflag']
    except Exception as err:
        print(err)
        return 422
    print("Query: %s"%json.dumps(nlquery).encode().decode("unicode-escape")) 
    vectors = v.vectorise(nlquery)
    print("Vectorisation length %d"%(len(vectors)))
    entities = p.link(vectors)
    # return json.dumps({'nlquery':d['nlquery'],'entities':entities}, indent=4, sort_keys=True).encode().decode("unicode-escape") 
    return json.dumps({'nlquery':d['nlquery'],'entities':entities}, indent=4, sort_keys=True)

@app.route('/nif', methods=['GET','POST'])
def processQueryNif():
    print("inside")
    content_format = request.headers.get('Content') or 'application/x-turtle'
    nif_body = request.data.decode("utf-8")
    print(nif_body)
    try:
        nif_doc = NIFCollection.loads(nif_body, format='turtle')
        #print(nif_doc)
        for context in nif_doc.contexts:
            vectors = v.vectorise(context.mention)
            entities = p.link(vectors)
            s = set()
            for idx,entityarr in entities.items():
                for ent in entityarr:
                    s.add(ent[0])
            for entity in s:
                context.add_phrase(beginIndex=0, endIndex=1, taIdentRef='http://www.wikidata.org/entity/'+entity)
        resp = Response(nif_doc.dumps())
        print(nif_doc.dumps())
        resp.headers['content-type'] = content_format
        return resp
    except Exception as e:
        print(e)
        return ''
    return ''


if __name__ == '__main__':
    http_server = WSGIServer(('', int(args.port)), app)
    http_server.serve_forever()
