
from __future__ import print_function
import sys,json,urllib, urllib2, re
from multiprocessing import Pool
import time

def hiturl(questionserial):
    question = questionserial[0]
    serial = questionserial[1]['id']
    req = urllib2.Request('http://localhost:4444/processQuery')
    req.add_header('Content-Type', 'application/json')
    try:
        inputjson = {'nlquery': question}
        start = time.time()
        response = urllib2.urlopen(req, json.dumps(inputjson))
        end = time.time()
        response = response.read()
        print(serial)
        print(question)
        return (serial,response,questionserial[1],end-start,len(question.split(' ')))
    except Exception as e:
        print(e)
        return(serial,'[]',questionserial[1],0,0)

f = open('simpleq_checked.json')
s = f.read()
d = json.loads(s)
f.close()
questions = []

for item in d:
    questions.append((item['question'],item))

pool = Pool(1)
responses = pool.imap(hiturl,questions)

_results = []

count = 0
totalentchunks = 0
tpentity = 0
fpentity = 0
fnentity = 0

for response in responses:
    count += 1
    _results.append((response[0],json.loads(response[1]),response[3],response[4]))


results = []
for result in _results:
    results.append(result)


print("total: %d" %count) 
f1 = open('simpleqtest_checked.json','w')
print(json.dumps(results),file=f1)
f1.close()
