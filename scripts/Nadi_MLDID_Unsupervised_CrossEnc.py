#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:58:14 2024

@author: vanikanjirangat
"""


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

from torch import nn
#We use the Bi-Encoder to encode all passages, so that we can use it with semantic search
model_path="UBC-NLP/MARBERTv2"
bi_encoder = SentenceTransformer(model_path)
bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_enc_path="nreimers/mmarco-mMiniLMv2-L12-H384-v1"
#cross-encoder/ms-marco-MiniLM-L-6-v2
cross_encoder=CrossEncoder(cross_enc_path, default_activation_function=nn.Sigmoid())

corpus_embeddings = bi_encoder.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
import pickle

import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
# Store sentences & embeddings on disc
import pickle
with open"./embeddings.pkl", "wb") as fOut:
    pickle.dump({"sentences": sentences, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
# Load sentences & embeddings from disc
import torch
with open(input_dir+"/embeddings.pkl", "rb") as f:
    #map_location=torch.device('cpu')
    #stored_data=torch.load(fIn,map_location=torch.device('cpu'))
    stored_data  = CPU_Unpickler(f).load()
    #stored_data = pickle.load(fIn,map_location='cpu', pickle_module=pickle)
    stored_sentences = stored_data["sentences"]
    stored_embeddings = stored_data["embeddings"]
    
print(len(stored_embeddings))
corpus_embeddings=stored_embeddings

# We also compare the results to lexical search (keyword search). Here, we use
# the BM25 algorithm which is implemented in the rank_bm25 package.

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np


# # We lower case our text and remove stop-words from indexing
# def bm25_tokenizer(text):
#     tokenized_doc = []
#     for token in text.lower().split():
#         token = token.strip(string.punctuation)

#         if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
#             tokenized_doc.append(token)
#     return tokenized_doc


# tokenized_corpus = []
# for passage in tqdm(sentences):
#     tokenized_corpus.append(bm25_tokenizer(passage))

tokenized_corpus=sentences
bm25 = BM25Okapi(tokenized_corpus)

# answer the query
from collections import Counter
bm25_results=[]
cross_enc_results=[]
enc_results=[]
def search(query):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    #bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    bm25_scores = bm25.get_scores(query)
    top_n = np.argpartition(bm25_scores, -10)[-10:]
    print("No. of items retrieved:",len(top_n))
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    print("bm25_hits:",bm25_hits)
    #df_bm25 = pd.DataFrame(bm25_hits)
    #bm25_results.append(bm25_hits)
    print("Top-3 lexical search (BM25) hits")
    c=[]
    for hit in bm25_hits[0:3]:
        print("\t{:.3f}\t{}\t{}\t{}".format(hit['score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]))
    for hit in bm25_hits:
      c.append([query,hit['score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]])
    labs=[x[-1] for x in c]
    sc=[x[1] for x in c]
    l=Counter(labs)
    l=l.most_common()
    #print(l)
    c1,c2,c3=[],[],[]
    for i in c:
        c1.append(i + [l])
    for i in c1:
        c2.append(i + [sc])
    for i in c2:
        c3.append(i + [labs])
        # c1.append(i+[sc])
        # c1.append(i+[labs])
    #c1=[l1.append(l) for l1 in c]
    #print(c1)
    bm25_results.append(c3)
    print(len(bm25_results))
    ##### Semantic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding
    #cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=10)
    #print("HITS:",hits)
    hits = hits[0]  # Get the hits for the first query
    print("HITS:",hits)



    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, sentences[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    for hit in hits[0:3]:
        print("\t{:.3f}\t{}\t{}\t{}".format(hit['score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]))

    d=[]
    for hit in hits[:10]:
      d.append([query,hit['score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]])
    labs=[x[-1] for x in d]
    sc=[x[1] for x in d]
    l=Counter(labs)
    l=l.most_common()
    #print(l)
    d1,d2,d3=[],[],[]
    for i in d:
        d1.append(i + [l])
    for i in d1:
        d2.append(i + [sc])
    for i in d2:
        d3.append(i + [labs])
    enc_results.append(d3)
    print(len(enc_results))
    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    #cross_enc_results.append(hits)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}\t{}\t{}".format(hit['cross-score'],sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]))
    e=[]
    for hit in hits[:10]:
      e.append([query,hit['cross-score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]])

    labs=[x[-1] for x in e]
    sc=[x[1] for x in e]
    l=Counter(labs)
    l=l.most_common()
    #print(l)
    e1,e2,e3=[],[],[]
    for i in e:
        e1.append(i + [l])
    for i in e1:
        e2.append(i + [sc])
    for i in e2:
        e3.append(i + [labs])

    cross_enc_results.append(e3)
    print(len(cross_enc_results))

path1="./subtask1_test_sentences.tsv"
import pandas as pd
path_test=path1
test=pd.read_csv(path_test,delimiter="\t")
queries=test['sentence'].values
print(len(queries))

from collections import Counter
for q in queries:

  search(query = q)
  print("################\n")
  
data = [v for lst in cross_enc_results for v in lst]
out_path="./"

df_enc_cross = pd.DataFrame(data, columns=['Query','Score','Sentence','Sent_Id','Label','Label_Counter','Score_List','Label_list'])
df_enc_cross_unique=df_enc_cross.drop_duplicates(subset='Query')
df_enc_cross.to_csv(out_path+"/df_enc_cross_Test.csv")
preds_cross=df_enc_cross_unique['Label_Counter'].values


def pred_counts(pred):
  preds=[]
  for i,item in enumerate(pred):
    p=[]
    j=0
    if len(item)<=3:# if the no. of labels predicted in top 10 <=3
      print(item)
      cl=[x[0] for x in item]
      p=[(y,'y') for y in cl]
    else:
      for m,k in enumerate(item):
        if m==0:
          if k[1]>=5:#5,6,7,8,9,10
            p.append((k[0],'y'))
            j=j+1
            while j<=len(item)-1:
              p.append((item[j][0],'n'))
              j=j+1
          elif k[1]<5 and k[1]>2:#4,3
            #print("&&",i)
            p.append((k[0],'y'))
            j=j+1
            while j<=len(item)-1:

              #print(j)
              if item[j][1]<=3 and item[j][1]>1:#3,2
                p.append((item[j][0],'y'))
              else:
                p.append((item[j][0],'n'))
              j=j+1

          else:
            p.append((k[0],'n'))
      #p=[list(i) for i in set(map(tuple, p))]
    preds.append(p)
  return preds

COUNTRIES = [
    "Algeria",
    "Bahrain",
    "Egypt",
    "Iraq",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Sudan",
    "Syria",
    "Tunisia",
    "UAE",
    "Yemen",
]

import numpy as np

l=COUNTRIES
def final_preds(preds):
  preds1=[]
  for q in preds:
    p1={}
    #for q in p:
    c1=[x[0] for x in q]
    #print(c1)
    c2=[x[1] for x in q]
    if len(c1)!=len(l):
      added = list(sorted(set(l) - set(c1)))
      print(added)

    for i,y in enumerate(c1):
      p1[y]=c2[i]
    for add in added:
      if add not in list(p1.keys()):
        p1[add]='n'
    #print(len(p1.keys()))
    preds1.append(p1)

  return preds1


preds_cross1=pred_counts(preds_cross)
preds1=final_preds(preds_cross1)
df_preds=pd.DataFrame(preds1)
df_preds = df_preds[COUNTRIES]

df_preds = df_preds.replace(np.nan, 0)
df_preds = df_preds.replace('y', 1)
df_preds = df_preds.replace('n', 0)
pred_labels=df_preds.values.tolist()
pred_labels1=[','.join((map(str,s))) for s in pred_labels]

with open (out_path+"./NLP_DI_subtask1_cross_Test","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")