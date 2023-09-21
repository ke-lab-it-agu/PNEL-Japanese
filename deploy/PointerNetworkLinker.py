from elasticsearch import Elasticsearch
import tensorflow as tf
import numpy as np
import pointer_net
import time,os,sys,json,re,requests
from Vectoriser import Vectoriser
import copy


class PointerNetworkLinker():
    def __init__(self, modelpath, rnnsize, attentionsize, layers):
        print("Initialising PointerNetworkLinker")
        self.modelpath = modelpath
        self.rnn_size = rnnsize
        self.attention_size = attentionsize
        self.num_layers = layers
        self.forward_only = True
        self.graph = tf.Graph()
        self.max_input_sequence_len = 3000
        self.max_output_sequence_len = 100
        self.beam_width = 1
        self.batch_size = 1
        self.forward_only = True
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.operation_timeout_in_ms=10000
            self.sess = tf.Session(config=config)
        self.build_model()
        print("Initialised PointerNetworkLinker")
    
    
    def build_model(self):
        with self.graph.as_default():
            self.model = pointer_net.PointerNet(batch_size=self.batch_size,
                        max_input_sequence_len=self.max_input_sequence_len,
                        max_output_sequence_len=self.max_output_sequence_len,
                        rnn_size=self.rnn_size,
                        attention_size=self.attention_size,
                        num_layers=self.num_layers,
                        beam_width=self.beam_width,
                        forward_only=self.forward_only)
            ckpt = tf.train.get_checkpoint_state(self.modelpath)
            print(ckpt, self.modelpath)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Load model parameters from %s" % ckpt.model_checkpoint_path)
                self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.sess.graph.finalize()

    def merge(self, entitytuple, entdict):
        span = entitytuple[1]
        tempentdict = copy.deepcopy(entdict)
        for k,v in tempentdict.items():
            for entchunk in v:
                entspan = entchunk[1]
                x = range(span[0],span[1]+1)
                y = range(entspan[0],entspan[1]+1)
                if len(set(x).intersection(y)) > 0:
                    if entitytuple in entdict[k]:
                        print("Exact already present, skip")
                        continue
                    print("Merging ", entitytuple, " into ", entdict)
                    entdict[k].append((entitytuple))
        

    def overlap(self,span,entdict):
        for k,v in entdict.items():
            for entchunk in v:
                entspan = entchunk[1]
                x = range(span[0],span[1]+1)
                y = range(entspan[0],entspan[1]+1)
                if len(set(x).intersection(y)) > 0:
                    print("Overlap exists between ",span," and ",entspan)
                    return True
        return False 
                    
    def processentities(self, entities):
        entdict = {}
        clustercount = 0
        for entitytuple in entities:
            entid,span, spanphrase, storedlabel = entitytuple
            if len(entdict) == 0:
                entdict[clustercount] = [entitytuple]
            else:
                if self.overlap(span, entdict):
                    self.merge(entitytuple,entdict)
                else:
                    clustercount += 1
                    entdict[clustercount] =  [entitytuple]
            print("entdict: ",entdict)
        return entdict

    def link(self, vectors):
        print("Entered pointer network linker ...")
        inputs = []
        self.outputs = []
        enc_input_weights = []
        dec_input_weights = []
        maxlen = 0
        questioninputs = []
        enc_input_len = len(vectors)
        if enc_input_len > self.max_input_sequence_len:
            print("Length too long, skip")
            return []
        for idx,word in enumerate(vectors):
            questioninputs.append(word[0])
        for i in range(self.max_input_sequence_len-enc_input_len):
            questioninputs.append([0]*1124)
        weight = np.zeros(self.max_input_sequence_len)
        weight[:enc_input_len]=1
        enc_input_weights.append(weight)
        inputs.append(questioninputs)
        self.test_inputs = np.stack(inputs)
        self.test_enc_input_weights = np.stack(enc_input_weights)
        predicted_ids,outputs = self.model.step(self.sess, self.test_inputs, self.test_enc_input_weights, update=False) 
        print("predicted_ids: ",list(predicted_ids[0][0]))
        entities = []
        for entnum in list(predicted_ids[0][0]):
            if entnum <= 0:
                continue
            #wordindex = vectors[entnum-1][0][1437]
            #if wordindex in seen:
            #    continue
            span = vectors[entnum-1][4] # [startindex, endindex]
            spanphrase = vectors[entnum-1][3] # of India
            storedlabel = vectors[entnum-1][2] # India
            entid = vectors[entnum-1][1] #Q668
            entities.append((entid,span, spanphrase, storedlabel))
            #print(vectors[entnum-1][0][1437], vectors[entnum-1][0][1438],vectors[entnum-1][0][1436], vectors[entnum-1][1], vectors[entnum-1][2], vectors[entnum-1][3], vectors[entnum-1][4])
        groupedentities = self.processentities(entities)
        print("predents: ",groupedentities)
        return groupedentities

if __name__ == '__main__':
    v = Vectoriser()
    vectors = v.vectorise("what electorate does anna bligh represent?")
    #p = PointerNetworkLinker("./models/data/webq1124/", 256, 64, 1)
    #entities = p.link(vectors)
    #p = PointerNetworkLinker("./models/data/lcq1124/", 512, 128, 1)
    #entities = p.link(vectors)
    #p = PointerNetworkLinker("./models/data/simpleq0/", 512, 128, 1)
    #entities = p.link(vectors) 

