import sys,os,json,re

gold = []
MissEntity = []

f = open('lcq2.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: x['id'])

for item in d:
    unit = {}
    unit['uid'] = item['id']
    unit['question'] = item['question']
    unit['entities'] = item['entities']
    unit['all'] = item
    gold.append(unit)

f = open('lcq2test.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: x[0])

tpentity = 0
fpentity = 0
fnentity = 0
tprelation = 0
fprelation = 0
fnrelation = 0
totalentchunks = 0
totalrelchunks = 0
mrrent = 0
mrrrel = 0
chunkingerror = 0
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
        totalentchunks += 1
        if goldentity in queryentities:
            tpentity += 1
        else:
            fnentity += 1
            miss = "false negative: " + queryitem[0] + ", " + goldentity
            MissEntity.append(miss)
    for queryentity in set(queryentities):
        if queryentity not in golditem['entities']:
            fpentity += 1
            miss = "false positive: " + queryitem[0] + " , " + goldentity
            MissEntity.append(miss)

precisionentity = tpentity/float(tpentity+fpentity)
recallentity = tpentity/float(tpentity+fnentity)
f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
print("precision entity = ",precisionentity)
print("recall entity = ",recallentity)
print("f1 entity = ",f1entity)

for Miss in MissEntity:
    f = open('outputMissEntity.txt', 'a')
    f.write(Miss)
    f.write("\n")
    f.close()