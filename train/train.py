import tensorflow as tf
import numpy as np
import pointer_net
import time
import os
import random
import sys
import json
import glob

tf.app.flags.DEFINE_integer("batch_size", 10,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 3000, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 100, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 512, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("models_dir", "", "Log directory")
tf.app.flags.DEFINE_string("data_path", "", "Training Data path.")
tf.app.flags.DEFINE_string("test_data_path", "", "Test Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "frequence to do per checkpoint.")
tf.app.flags.DEFINE_integer("epoch_limit", 10, "stop after these many epochs")

FLAGS = tf.app.flags.FLAGS

# flag = 0
# pre = 0
# recall = 0
# f1 = 0
# MissEntity = []

class EntityLinker(object):
  def __init__(self,forward_only,trainfiles):
    self.forward_only = forward_only
    self.epoch = 0
    self.bestf1 = 0
    self.graph = tf.Graph()
    self.testgraph = tf.Graph()
    self.trainfiles = trainfiles
    with self.graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth=True
      self.sess = tf.Session(config=config)
    with self.testgraph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.operation_timeout_in_ms=6000
      self.testsess = tf.Session(config=config)
    #self.read_test_data()
    testlinecount = 0
    self.build_model()
    with open(FLAGS.test_data_path) as rfp:
      for line in rfp:
        testlinecount += 1
    print(testlinecount, " lines in test file")
    random.seed(1)
    self.randomtestlinenumbers = random.sample(range(1,testlinecount-1),100)
    print("Will test the following line numbers: ",self.randomtestlinenumbers)


  def read_data(self, step):
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []
    maxlen = 0
    linecount = 0
    x = random.randint(0,len(self.trainfiles)-1) 
    with open(self.trainfiles[x],'r') as fp:
      print(step,self.trainfiles[x])
      for line in fp:
        seen = []
        linecount += 1
        line = line.strip()
        try:
            question = json.loads(line)
        except Exception as e:
            print(e)
            continue
        questioninputs = []
        questionoutputs = []
        for idx,word in enumerate(question[2]):
          questioninputs.append(word[0])
          if word[2] == 1.0:
            questionoutputs.append(idx+1)
        enc_input_len = len(question[2]) 
        if enc_input_len > FLAGS.max_input_sequence_len:
          continue
        for i in range(FLAGS.max_input_sequence_len-enc_input_len):
          questioninputs.append([0]*1124)
        weight = np.zeros(FLAGS.max_input_sequence_len)
        weight[:enc_input_len]=1
        enc_input_weights.append(weight)
        inputs.append(questioninputs)
        output=[pointer_net.START_ID]
        for i in questionoutputs:
          # Add 2 to value due to the sepcial tokens
          output.append(int(i)+2)
        output.append(pointer_net.END_ID)
        dec_input_len = len(output)-1
        output += [pointer_net.PAD_ID]*(FLAGS.max_output_sequence_len-dec_input_len)
        output = np.array(output)
        outputs.append(output)
        weight = np.zeros(FLAGS.max_output_sequence_len)
        weight[:dec_input_len]=1
        dec_input_weights.append(weight)
    if len(inputs) < FLAGS.batch_size:
      return False

    self.inputs = np.stack(inputs)
    self.enc_input_weights = np.stack(enc_input_weights)
    self.outputs = np.stack(outputs)
    self.dec_input_weights = np.stack(dec_input_weights)
#    print("Load inputs:            " +str(self.inputs.shape))
#    print("Load enc_input_weights: " +str(self.enc_input_weights.shape))
#    print(self.enc_input_weights[0])
#    print("Load outputs:           " +str(self.outputs.shape))
#    print(self.outputs)
#    print("Load dec_input_weights: " +str(self.dec_input_weights.shape))
#    print(self.dec_input_weights)


  def get_batch(self, step):
    self.read_data(step)
    return self.inputs,self.enc_input_weights,\
      self.outputs, self.dec_input_weights

  def get_test_batch(self):
    return self.test_inputs,self.test_enc_input_weights,\
      self.test_outputs, self.test_dec_input_weights

  def build_model(self):
    with self.testgraph.as_default():
      self.testmodel = pointer_net.PointerNet(batch_size=1,
                    max_input_sequence_len=FLAGS.max_input_sequence_len,
                    max_output_sequence_len=FLAGS.max_output_sequence_len,
                    rnn_size=FLAGS.rnn_size,
                    attention_size=FLAGS.attention_size,
                    num_layers=FLAGS.num_layers,
                    beam_width=FLAGS.beam_width,
                    learning_rate=FLAGS.learning_rate,
                    max_gradient_norm=FLAGS.max_gradient_norm,
                    forward_only=True)
    with self.graph.as_default():
      # Build model
      self.model = pointer_net.PointerNet(batch_size=FLAGS.batch_size, 
                    max_input_sequence_len=FLAGS.max_input_sequence_len, 
                    max_output_sequence_len=FLAGS.max_output_sequence_len, 
                    rnn_size=FLAGS.rnn_size, 
                    attention_size=FLAGS.attention_size, 
                    num_layers=FLAGS.num_layers,
                    beam_width=FLAGS.beam_width, 
                    learning_rate=FLAGS.learning_rate, 
                    max_gradient_norm=FLAGS.max_gradient_norm, 
                    forward_only=self.forward_only)
      self.sess.run(tf.global_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(FLAGS.models_dir)
      print(ckpt, FLAGS.models_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
      self.writer = tf.summary.FileWriter(FLAGS.models_dir + '/train',self.sess.graph)


  def train(self):
    step_time = 0.0
    loss = 0.0
    valloss = 0.0
    current_step = 0
    test_step_loss = 0.0
    besttestloss = 99999
    gstep = None
    while True:
      start_time = time.time()
      inputs,enc_input_weights, outputs, dec_input_weights = \
                  self.get_batch(current_step)
      if current_step >= len(self.trainfiles)-1:
        print(current_step)
        current_step = 0
        self.epoch += 1
        self.testall()
        if self.epoch > FLAGS.epoch_limit:
          # global flag
          # global pre
          # global recall
          # global f1
          # print("total precision  %f  total recall %f  total f1 %f  "%(pre/flag, recall/flag, f1/flag))
          print("%d epochs done, quit"%FLAGS.epoch_limit)
          
          # global MissEntity
          # f = open('outputMissEntity.txt', 'x', encoding='UTF-8')
          # f.writelimes(MissEntity)
          # f.close()
          
          sys.exit(1)
        continue
      if inputs.shape[0] < FLAGS.batch_size:
        print("less than batch size")
        current_step += 1
        continue
      # for i in inputs:
      #   print(len(i[0]),len(i[0]))
      # exit(1)
      summary, step_loss, predicted_ids_with_logits, targets, debug_var, rnn_output, llogits = \
              self.model.step(self.sess, inputs, enc_input_weights, outputs, dec_input_weights, update=True)
      # print("steploss: ",step_loss)
      # print("predicted ids with logits:",  predicted_ids_with_logits)
      # print("targets: ", targets) 
      # print("debug-var: ",debug_var)
      # print("rnnout: ", rnn_output)
      # print("rnnoutshape: ", rnn_output.shape)
      # print("llogits: ",llogits)
      # print("llogitsshape: ",llogits.shape)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      current_step += 1
      loss += step_loss# / FLAGS.steps_per_checkpoint
      with self.sess.as_default():
        gstep = self.model.global_step.eval()
  
      #Time to print statistic and save model
      if gstep % FLAGS.steps_per_checkpoint == 0:
        loss /= float(FLAGS.steps_per_checkpoint)
        print ("global step %d step-time %.2f loss %.2f epoch %d" % (gstep, step_time, loss, self.epoch))
        #Write summary
        self.writer.add_summary(summary, gstep)
        checkpoint_path = os.path.join(FLAGS.models_dir, "convex_hull.ckpt")
        self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        self.testall()
        step_time, loss = 0.0, 0.0

  def run(self):
    self.train()

  def getvector(self,d):
    inputs = []
    enc_input_weights = []
    dec_input_weights = []
    maxlen = 0
    self.testoutputs = []
    for question in d:
      questioninputs = []
      enc_input_len = len(question[2])
      #print(enc_input_len)
      if enc_input_len > FLAGS.max_input_sequence_len:
        print("Length too long, skip")
        continue
      for idx,word in enumerate(question[2]):
        questioninputs.append(word[0])
      for i in range(FLAGS.max_input_sequence_len-enc_input_len):
        questioninputs.append([0]*1124)
    self.testoutputs.append(question[1])
    weight = np.zeros(FLAGS.max_input_sequence_len)
    weight[:enc_input_len]=1
    enc_input_weights.append(weight)
    inputs.append(questioninputs)
    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)

  def calculatef1(self, batchd, predictions, tp,fp,fn):
    # global MissEntity
        
    for inputquestion,prediction,groundtruth in zip(batchd, predictions, self.testoutputs):
      idtoentity = {}
      predents = set()
      gtents = groundtruth
      #print(len(self.test_inputs))
      for entnum in list(prediction[0]):
        if entnum <= 0:
          continue
        predents.add(inputquestion[2][entnum-1][1])
      print(gtents,predents)
      for goldentity in gtents:
        #totalentchunks += 1
        if goldentity in predents:
          tp += 1
        else:
          fn += 1
          # missquestion = "false negative: " + inputquestion 
          # print(missquestion)
          # MissEntity.append(missquestion)
      for queryentity in predents:
        if queryentity not in gtents:
          fp += 1
          # missquestion = "false positive: " + inputquestion 
          # MissEntity.append(missquestion)
    try:
      precisionentity = tp/float(tp+fp)
      recallentity = tp/float(tp+fn)
      f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
      #print("precision entity = ",precisionentity)
      #print("recall entity = ",recallentity)
      #print("f1 entity = ",f1entity)
    except Exception as e:
      #print(e)
      pass
    return tp,fp,fn
    #print(tp,fp,fn)

  def testall(self):
    print("Test set evaluation running ...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.models_dir)
    print(ckpt, FLAGS.models_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      print("Load model parameters from %s" % ckpt.model_checkpoint_path)
      self.testmodel.saver.restore(self.testsess, ckpt.model_checkpoint_path)
    tp = 0
    fp = 0
    fn = 0
    linecount = 0
    batchd = []
    with open(FLAGS.test_data_path) as rfp:
      for line in rfp:
        linecount += 1
        if linecount not in self.randomtestlinenumbers:
          continue
        line = line.strip()
        d = json.loads(line)
        if len(d) > FLAGS.max_input_sequence_len:
          print("Skip question, too long")
          continue
  #      #print(len(d))
        batchd.append(d)
        #print(linecount)
        try:
          self.getvector(batchd)
          predicted,_ = self.testmodel.step(self.testsess, self.test_inputs, self.test_enc_input_weights, update=False)
          _tp,_fp,_fn = self.calculatef1(batchd,predicted,tp,fp,fn)
          batchd = []
        except Exception as e:
          #print(e)
          batchd = []
          continue
        tp = _tp
        fp = _fp
        fn = _fn
    precisionentity = tp/float(tp+fp+0.001)
    recallentity = tp/float(tp+fn+0.001)
    f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity+0.001)
    print("precision  %f  recall %f  f1 %f  globalstep %d  epoch %d"%(precisionentity, recallentity, f1entity, self.model.global_step.eval(session=self.sess), self.epoch))
    # global flag
    # global pre
    # global recall
    # global f1
    # flag = flag + 1
    # pre = pre + precisionentity
    # recall = recall + recallentity
    # f1 = f1 + f1entity
    if f1entity > self.bestf1:
      self.bestf1 = f1entity
      print("Best f1 so far, saving ...")
      checkpoint_path = os.path.join(FLAGS.models_dir+'/solid/', "%f_%d_%d_convex_hull.ckpt"%(f1entity,self.model.global_step.eval(session=self.sess),self.epoch))
      self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step.eval(session=self.sess))
      #save model now 

def main(_):
  trainfiles = glob.glob(FLAGS.data_path+'/*')
  entitylinker = EntityLinker(FLAGS.forward_only,trainfiles)
  entitylinker.run()

if __name__ == "__main__":
  tf.app.run()
