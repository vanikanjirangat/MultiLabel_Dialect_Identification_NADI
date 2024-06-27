# -*- coding: utf-8 -*-
"@author: vanikanjirangat"

from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=1, lora_dropout=0.1
)

from peft import get_peft_model

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
le = LabelEncoder()
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from collections import Counter

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)



root_dir = "./"
input_dir="./"



from time import perf_counter, sleep,process_time


#ENCODER MODELS

import time
import datetime
from time import perf_counter, sleep,process_time
from transformers import AutoModel, AutoTokenizer, AutoConfig,AutoModelForSequenceClassification
class Model:
    def __init__(self,path,model_name):
        # self.args = args
        self.path=path
        self.MAX_LEN=128
        self.model_name=model_name
        print("Running %s"%(self.model_name))
        num_labels=2
        #num_labels=21
        if self.model_name=="arabert":
          #mod_path='aubmindlab/bert-base-arabert'
          #mod_path='aubmindlab/bert-base-arabertv2'
         # mod_path='aubmindlab/bert-base-arabertv02-twitter'
          mod_path="aubmindlab/bert-large-arabertv02-twitter"
        elif self.model_name=="multidialect":
          mod_path='bashar-talafha/multi-dialect-bert-base-arabic'
        elif self.model_name=="multibert":
          mod_path='bert-base-multilingual-cased'
        elif self.model_name=="camelbert":
          mod_path="CAMeL-Lab/bert-base-arabic-camelbert-da"
        elif self.model_name=="marbert":
          mod_path="UBC-NLP/MARBERT"
        elif self.model_name=="marbertv2":
          mod_path="UBC-NLP/MARBERTv2"
          #load MAEBERT model from huggingface

          #mod_path="CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus26"

        # elif self.model_name=="arat5":
        #   mod_path="UBC-NLP/AraT5v2-base-1024"
        # self.tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        self.tokenizer = AutoTokenizer.from_pretrained(mod_path,hidden_dropout_prob=0.3)
        self.config = AutoConfig.from_pretrained(mod_path,num_labels=num_labels)



    def extract_data(self,name,XY=None):
        file =self.path+name
s	self.name=name
        df = pd.read_csv(file)
        df.replace(np.nan,'NIL', inplace=True)
        sentences= df["#2_content"].values
        labels = df["#3_label"].values
        #sentences= df["sentence"].values
        #labels = df["label"].values
        return (sentences,labels)
    def extract_data_test(self,name,XY=None):
      file =self.path+name
      df = pd.read_csv(file, delimiter='\t')
      df.replace(np.nan,'NIL', inplace=True)
      sentences= df["#2_content"].values
      #labels = df["#3_label"].values
      self.sentences=sentences
      return sentences


    def process_inputs(self,sentences,labels):
      sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      # sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      # torch_idx = torch.tensor(sentence_idx)
      tags_vals = list(labels)
      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      labels=le.fit_transform(labels)

      print('MAPPINGS:',le_name_mapping)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)


      # token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      # token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs, labels = input_ids, labels
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      # validation_masks = torch.tensor(validation_masks).to(torch.int64)
      # self.types=torch.tensor(types).to(torch.int64)
      self.data = TensorDataset(self.inputs,self.masks, self.labels)
      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=32)

      # return (self.inputs,self.labels,self.masks,self.types)
    def process_dev_inputs(self,sentences,labels):
      sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      torch_idx = torch.tensor(sentence_idx)
      tags_vals = list(labels)
      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      labels=le.fit_transform(labels)

      print("DEV_MAPPING:",le_name_mapping)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)


      # token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      # token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs, labels = input_ids, labels
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      # validation_masks = torch.tensor(validation_masks).to(torch.int64)
      # self.types=torch.tensor(types).to(torch.int64)
      self.data = TensorDataset(self.inputs,self.masks, self.labels)
      self.sampler = RandomSampler(self.data)
      self.validationdataloader = DataLoader(self.data, sampler=self.sampler, batch_size=32)

    def process_inputs_test_official(self,sentences,batch_size=1):
      sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      torch_idx = torch.tensor(sentence_idx)
      # tags_vals = list(labels)
      # le.fit(labels)
      # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      # labels=le.fit_transform(labels)

      #print(le_name_mapping)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)


      # token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      # token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs = input_ids
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      #self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      #self.act_ids = torch.tensor(act_ids).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      #self.torch_idx = torch.tensor(sentence_idx).to(torch.int64)
      self.data = TensorDataset(self.inputs,self.masks)
      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=batch_size)    


    def process_inputs_test(self,sentences,labels,act_ids,batch_size=1):
      sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      torch_idx = torch.tensor(sentence_idx)
      tags_vals = list(labels)
      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      labels=le.fit_transform(labels)

      print(le_name_mapping)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)


      # token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      # token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs, labels = input_ids, labels
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.act_ids = torch.tensor(act_ids).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      self.torch_idx = torch.tensor(sentence_idx).to(torch.int64)
      self.data = TensorDataset(self.inputs,self.masks, self.labels,self.torch_idx,self.act_ids)
      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=batch_size)

    def train_save_load(self,train=1,retrain=0,label_smoothing = -1,LoRA=0,XY=None):
      epochs=5
      OUTPUT_DIR = input_dir
      #OUTPUT_DIR = input_dir+"/NADI2021/"
      #WEIGHTS_NAME = "NADI2023_%stwitterL.bin"%(self.model_name)
      #output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      if self.model_name=="arabert":
        #mod_path='aubmindlab/bert-base-arabert'
        #mod_path='aubmindlab/bert-base-arabertv2'
        #mod_path='aubmindlab/bert-base-arabertv02-twitter'
        mod_path="aubmindlab/bert-large-arabertv02-twitter"
      elif self.model_name=="multidialect":
        mod_path='bashar-talafha/multi-dialect-bert-base-arabic'
      elif self.model_name=="multibert":
        mod_path='bert-base-multilingual-cased'
      elif self.model_name=="camelbert":
          mod_path="CAMeL-Lab/bert-base-arabic-camelbert-da"
      elif self.model_name=="marbert":
        mod_path="UBC-NLP/MARBERT"
      elif self.model_name=="marbertv2":
          mod_path="UBC-NLP/MARBERTv2"
          #mod_path="CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus26"
      # elif self.model_name=="arat5":
      #     mod_path="UBC-NLP/AraT5v2-base-1024"
      mp=mod_path.split("/")[-1]


      print("training starts...")
      if LoRA:
        print("LoRA")
        self.model = AutoModelForSequenceClassification.from_pretrained(mod_path,config=self.config)
        self.model = get_peft_model(self.model, lora_config)
        WEIGHTS_NAME = "NADI2023_%s%sLoRA8.bin"%(mp,epochs)
        output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      if retrain!=1:
        self.model = AutoModelForSequenceClassification.from_pretrained(mod_path,config=self.config)
        WEIGHTS_NAME = "NADI2023_%s%s0.3.bin"%(mp,epochs)
        output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)

      else:
        # self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(mod_path,config=self.config)
        state_dict = torch.load(output_model_file)
        self.model.load_state_dict(state_dict)
        WEIGHTS_NAME = "NADI2024_%sBinary.bin"%(self.name)

        OUTPUT_DIR = input_dir
        output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      self.model.cuda()
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                                                                                                                                     'weight_decay_rate': 0.0}]
      #optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
      #optimizer = AdamW(optimizer_grouped_parameters,lr=5e-5)
      #optimizer = AdamW(optimizer_grouped_parameters,lr=3e-5)
      optimizer = AdamW(optimizer_grouped_parameters,lr=1e-5)

      train_loss_set = []
      out_train={}
      #batch_size=32
      true=[]
      logits_all=[]
      output_dicts = []
      batch_size=8
      #epochs = 4
      #import time
      #start_time = time.time()
      start = perf_counter()
      start1=process_time()
      start2=time.time()


      if train==1:
        for _ in trange(epochs, desc="Epoch"):
          # Trainin
          # Set our model to training mode (as opposed to evaluation mode
          self.model.train()
          # Tracking variables
          tr_loss = 0
          nb_tr_examples, nb_tr_steps = 0, 0
          # Train the data for one epoch
          for step, batch in enumerate(self.dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids,b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            # loss = model(b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask, labels=b_labels)
            loss,logits= self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,return_dict=False)
            if label_smoothing == -1:
              logits=logits
            # else:
            #   criterion = LabelSmoothingLoss(label_smoothing)
            #   loss=criterion(logits,b_labels)




            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
          print("Train loss: {}".format(tr_loss/nb_tr_steps))
        #print("--- %s seconds ---" % (time.time() - start_time))
        end = perf_counter()
        end1=process_time()
        end2=time.time()

        print(f"Time taken to execute code, perf_counter : {end-start}")
        print(f"Time taken to execute code, process_time : {end1-start1}")
        print(f"Time taken to execute code, time : {end2-start2}")
        torch.save(self.model.state_dict(), output_model_file)

      else:
        state_dict = torch.load(output_model_file)
        self.model.load_state_dict(state_dict)
      return output_dicts

    def eval(self,label_smoothing = -1):
      batch_size=8
      eval_loss = 0
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables
      self.predictions , self.true_labels = [], []
      output_dicts=[]


      for batch in self.dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          if label_smoothing == -1:
            logits=outputs[0]
            loss=outputs[1]
          # else:
          #   criterion = LabelSmoothingLoss(label_smoothing)
          #   loss=criterion(logits,b_labels)

          eval_loss += loss.item()
          self.dataloader.set_description(f'eval loss = {(eval_loss / i):.6f}')
      return eval_loss / len(self.dataloader)


    def simple_test(self):
      batch_size=8
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables
      self.predictions , self.true_labels = [], []
      output_dicts=[]

      start = perf_counter()
      start1=process_time()
      start2=time.time()


      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels = batch
        #print(b_labels)
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits=outputs[0]
          for j in range(logits.size(0)):
            probs = F.softmax(logits[j], -1)
            output_dict = {
                # 'index': batch_size * i + j,
                'true': b_labels[j].cpu().numpy().tolist(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
            }
            output_dicts.append(output_dict)
      end = perf_counter()
      end1=process_time()
      end2=time.time()


      y_true = [output_dict['true'] for output_dict in output_dicts]
      y_pred = [output_dict['pred'] for output_dict in output_dicts]
      y_conf = [output_dict['conf'] for output_dict in output_dicts]
      #print(y_true)
      #print(y_pred)

      accuracy = accuracy_score(y_true, y_pred) * 100.
      f1 = f1_score(y_true, y_pred, average='macro') * 100.
      confidence = np.mean(y_conf) * 100.

      results_dict = {
          'accuracy': accuracy_score(y_true, y_pred) * 100.,
          'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
          'confidence': np.mean(y_conf) * 100.,
      }
      print(results_dict)
      print(classification_report(y_true,y_pred))
      print(confusion_matrix(y_true,y_pred))
      print(f"Time taken to infer, perf_counter : {end-start}")
      print(f"Time taken to infer, process_time : {end1-start1}")
      print(f"Time taken to infer, time : {end2-start2}")
      return output_dicts

    def test(self,sents):
      batch_size=8
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables
      self.predictions , self.true_labels,self.sents,self.actsents = [], [],[],[]
      output_dicts=[]


      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels,b_index,b_ids = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits=outputs[0]


          for j in range(logits.size(0)):
            probs = F.softmax(logits[j], -1)
            output_dict = {
                # 'index': batch_size * i + j,
                'true': b_labels[j].cpu().numpy().tolist(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
                'actsent_ids'   : b_ids[j].cpu().numpy().tolist(),
                'sent_ids'   : b_index[j].cpu().numpy().tolist(),
                'sents' : sents[b_index[j]]
            }
            output_dicts.append(output_dict)
      y_true = [output_dict['true'] for output_dict in output_dicts]
      y_pred = [output_dict['pred'] for output_dict in output_dicts]
      y_conf = [output_dict['conf'] for output_dict in output_dicts]

      accuracy = accuracy_score(y_true, y_pred) * 100.
      f1 = f1_score(y_true, y_pred, average='macro') * 100.
      confidence = np.mean(y_conf) * 100.

      results_dict = {
          'accuracy': accuracy_score(y_true, y_pred) * 100.,
          'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
          'confidence': np.mean(y_conf) * 100.,
      }
      print(results_dict)
      print(classification_report(y_true,y_pred))
      print(confusion_matrix(y_true,y_pred))
      # print("writing the results...")
      # class_map={'algeria': 0, 'bahrain': 1, 'egypt': 2, 'iraq': 3, 'jordan': 4, 'ksa': 5, 'kuwait': 6, 'lebanon': 7, 'libya': 8, 'morocco': 9, 'oman': 10, 'palestine': 11, 'qatar': 12, 'sudan': 13, 'syria': 14, 'tunisia': 15, 'uae': 16, 'yemen': 17}
      # c={}
      # for i in class_map:
      #   c[class_map[i]]=i
      # p=[]
      # a=[]
      # for i,k in enumerate(y_pred):
      #     p.append(c[k])
      #     a.append(c[y_true[i]])
      # path=input_dir+"/results/AraBERTTestA.txt"
      # gold_path=input_dir+"/results/AraBERTgold.txt"
      # with open(path,"w") as f:
      #   for r in p:
      #     f.write(str(r))
      #     f.write("\n")
      # with open(gold_path,"w") as f:
      #   for r in a:
      #     f.write(str(r))
      #     f.write("\n")
      return output_dicts
    def official_test(self):
      batch_size=1
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables
      self.predictions = []
      output_dicts=[]

      start = perf_counter()
      start1=process_time()
      start2=time.time()
      
      

      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask= batch
        #print(b_labels)
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits=outputs[0]
          for j in range(logits.size(0)):
            probs = F.softmax(logits[j], -1)
            output_dict = {
                # 'index': batch_size * i + j,
                #'true': b_labels[j].cpu().numpy().tolist(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
            }
            output_dicts.append(output_dict)
      end = perf_counter()
      end1=process_time()
      end2=time.time()


      #y_true = [output_dict['true'] for output_dict in output_dicts]
      y_pred = [output_dict['pred'] for output_dict in output_dicts]
      y_conf = [output_dict['conf'] for output_dict in output_dicts]

      # accuracy = accuracy_score(y_true, y_pred) * 100.
      # f1 = f1_score(y_true, y_pred, average='macro') * 100.
      # confidence = np.mean(y_conf) * 100.

      # results_dict = {
      #     'accuracy': accuracy_score(y_true, y_pred) * 100.,
      #     'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
      #     'confidence': np.mean(y_conf) * 100.,
      # }
      # print(results_dict)
      # print(classification_report(y_true,y_pred))
      # print(confusion_matrix(y_true,y_pred))
      print("writing the results...")
      maps= {'Algeria': 0, 'Bahrain': 1, 'Egypt': 2, 'Iraq': 3, 'Jordan': 4, 'Kuwait': 5, 'Lebanon': 6, 'Libya': 7, 'Morocco': 8, 'Oman': 9, 'Palestine': 10, 'Qatar': 11, 'Saudi_Arabia': 12, 'Sudan': 13, 'Syria': 14, 'Tunisia': 15, 'UAE': 16, 'Yemen': 17}
      #c={}
      maps = {v: k for k, v in maps.items()}
      # for i in class_map:
      #   c[class_map[i]]=i
      # p=[]
      # a=[]
      # for i,k in enumerate(y_pred):
      #     p.append(c[k])
          #a.append(c[y_true[i]])
      # path=input_dir+"/results/ELDI_MARBERTV2.txt"
      
        #gold_path=input_dir+"/results/AraBERTgold.txt"
      print(maps)
      with open(input_dir+"/ELDI_LORA_MARBERTV2.txt", 'w') as f:
        for p in y_pred:
          f.write(maps[p])
          f.write("\n")
      # with open(path,"w") as f:
      #   for r in p:
      #     f.write(str(r))
      #     f.write("\n")
      # with open(gold_path,"w") as f:
      #   for r in a:
      #     f.write(str(r))
      #     f.write("\n")
      return output_dicts
    # def test_inc(self,batch_size=1):
    #   #batch_size : should be no. of fragments after incremental processing
    #   # Put model in evaluation mod
    #   self.model.eval()
    #   # Tracking variables
    #   self.predictions , self.true_labels = [], []
    #   output_dicts=[]


    #   for batch in self.dataloader:
    #     # Add batch to GPU
    #     batch = tuple(t.to(device) for t in batch)
    #     # Unpack the inputs from our dataloader
    #     b_input_ids,b_input_mask, b_labels = batch
    #     # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    #     with torch.no_grad():
    #       # Forward pass, calculate logit predictions
    #       outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    #       logits=outputs[0]
    #       for j in range(logits.size(0)):
    #         probs = F.softmax(logits[j], -1)
    #         output_dict = {
    #             # 'index': batch_size * i + j,
    #             'true': b_labels[j].cpu().numpy().tolist(),
    #             'pred': logits[j].argmax().item(),
    #             'conf': probs.max().item(),
    #             'logits': logits[j].cpu().numpy().tolist(),
    #             'probs': probs.cpu().numpy().tolist(),
    #         }
    #         output_dicts.append(output_dict)
    #   y_true = [output_dict['true'] for output_dict in output_dicts]
    #   y_pred = [output_dict['pred'] for output_dict in output_dicts]
    #   y_conf = [output_dict['conf'] for output_dict in output_dicts]

    #   accuracy = accuracy_score(y_true, y_pred) * 100.
    #   f1 = f1_score(y_true, y_pred, average='macro') * 100.
    #   confidence = np.mean(y_conf) * 100.

    #   results_dict = {
    #       'accuracy': accuracy_score(y_true, y_pred) * 100.,
    #       'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
    #       'confidence': np.mean(y_conf) * 100.,
    #   }
    #   print(results_dict)
    #   print(classification_report(y_true,y_pred))
    #   print(confusion_matrix(y_true,y_pred))
    #   return output_dicts

#input_dir
path="./NADI-2024/NADI2023_Subtask1_TRAIN.tsv"
import pandas as pd
import numpy as np
def extract_data(name):
    file = name
    df = pd.read_csv(file, delimiter='\t')
    df.replace(np.nan,'NIL', inplace=True)
    sentences= df["#2_content"].values
    labels = df["#3_label"].values
    #sentences= df["#2_tweet"].values
    #labels = df["#3_country_label"].values
    return (sentences,labels)
sentences,labels=extract_data(path)
class_names=list(set(labels))

pos={}
negs={}

import random
for i, lab in enumerate(class_names):
  samples=[]
  print("Class is %s"%(lab))
  pos[lab]=[(sent,labels[j])for j, sent in enumerate(sentences) if labels[j]==lab]

  n=[(sent,labels[j])for j, sent in enumerate(sentences) if labels[j]!=lab]
  #print(n[:10])
  random.shuffle(n)
  #print(n[:10])



  negs[lab]=random.sample(n,1000)
  print(len(pos[lab]),len(negs[lab]))
  samples=pos[lab]
  samples.extend(negs[lab])
  print(len(samples))
  samples_mod=[(x[0],1) if x[1]==lab else (x[0],0) for x in samples]
  print(len(samples_mod))
  df = pd.DataFrame(samples_mod, columns =['sentence', 'label'])
  df.to_csv("./NADI-2024/%s_binary.tsv"%(lab))
  
  
path="./NADI-2024/"

#path=input_dir+"/NADI2021/Subtask_1.1+2.1_MSA/"#2021

#path

#m = Model(path,"arabert")
#m = Model(path,"multidialect")
#m = Model(path,"camelbert")
#m = Model(path,"multibert")
#m=Model(path,"arat5")
#m = Model(path,"marbert")
m = Model(path,"marbertv2")
XY=0

# sentences_dev,labels_dev=m.extract_data('/dev.txt',XY=0)
for i, lab in enumerate(class_names): 
    print("Training the %s class"%(lab))
    sentences_train,labels_train=m.extract_data("./NADI-2024/%s_binary.tsv"%(lab),XY=0)


    
    print(len(sentences_train),len(labels_train))
    
    print("TRAIN STATISTICS")
    Counter(labels_train)
    # TV=0#use only training
    # if TV:
    #   sentences_train=np.append(sentences_train,sentences_dev)
    
    #   labels_train=np.append(labels_train,labels_dev)

    print(len(sentences_train),len(labels_train))
    
    m.process_inputs(sentences_train,labels_train)

    #print("Training the model MARBERTV2 model with drop out=0.3 and lr=1e-5 with 10 epochs")
    out=m.train_save_load(train=0,retrain=0,label_smoothing = -1,LoRA= 1, XY=0)#TRAIN the model#
    print("Trained..")



#INSTRUCTION TUNING
#n-gram tuning--?

#MARBERTV2 finetuned drop out=0.3 and lr=1e-5, batch=8,epochs=10
# act_out_dev=m.simple_test()
# sentences_test=m.extract_data_test('/NADI2023_Subtask1_TEST_Unlabeled.tsv',XY=0)
# m.process_inputs_test_official(sentences_test)
# act_out_test=m.official_test()


