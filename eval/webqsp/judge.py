import sys,os,json,re

gold = []
MissEntity = []

f = open('input/webqsp.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: x['id'])

for item in d:
    unit = {}
    unit['uid'] = item['id']
    unit['question'] = item['question']
    unit['entities'] = item['entities']
    unit['all'] = item
    gold.append(unit)

f = open('result.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: x[0])

tpentity = 0
fpentity = 0
fnentity = 0
for queryitem,golditem in zip(d,gold):
    if queryitem[0] != golditem['uid']:
        print(queryitem[0], golditem['uid'])
        print('uid mismatch')
        sys.exit(1)
    queryentities = []
    if 'entities' in queryitem[1]:
        if len(queryitem[1]['entities']) > 0:
            for k,v in queryitem[1]['entities'].iteritems():
                queryentities.append(v[0][0])
    print(set(golditem['entities']),set(queryentities), golditem['question'])
    if None in set(golditem['entities']):
        print('skip none')
        continue
    for goldentity in set(golditem['entities']):
        if goldentity == None:
            print("skip none")
            continue
        if goldentity in queryentities:
            tpentity += 1
        else:
            fnentity += 1
            miss = "false negative: " + queryitem[0] + ", " + goldentity
    for queryentity in set(queryentities):
        if queryentity not in golditem['entities']:
            fpentity += 1
            miss = "false positive: " + queryitem[0] + " , " + goldentity

precisionentity = tpentity/float(tpentity+fpentity)
recallentity = tpentity/float(tpentity+fnentity)
f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
print("precision entity = ",precisionentity)
print("recall entity = ",recallentity)
print("f1 entity = ",f1entity)