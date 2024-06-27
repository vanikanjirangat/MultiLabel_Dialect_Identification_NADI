# -*- coding: utf-8 -*-
"@author: vanikanjirangat"

import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)


root_dir = "./"
input_dir="./"

path="./NADI2023_Subtask1_TRAIN.tsv"

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

print(len(sentences),len(labels))

list(set(labels))


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
#cross_encoder = CrossEncoder(cross_enc_path)

# # As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# # about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

# wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

# if not os.path.exists(wikipedia_filepath):
#     util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

# passages = []
# with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
#     for line in fIn:
#         data = json.loads(line.strip())

#         #Add all paragraphs
#         #passages.extend(data['paragraphs'])

#         #Only add the first paragraph
#         passages.append(data['paragraphs'][0])

# print("Passages:", len(passages))

# We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)

import re
from nltk import word_tokenize
def removeNonArabicChar(text):
    return re.sub(r'\s*[A-Za-z0-9]+\b', '', text)
import nltk
nltk.download('punkt')

#sentences=[removeNonArabicChar(x) for x in sentences]

#sentences=sentences1
corpus_embeddings = bi_encoder.encode(sentences, convert_to_tensor=True, show_progress_bar=True)


import pickle

# with open(input_dir+"/P_embeddings.pkl", "wb") as fOut:#preprocessed_embeddings
#     pickle.dump({"sentences": sentences, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# # Store sentences & embeddings on disc

# with open(input_dir+"/partial_embeddings.pkl", "wb") as fOut:
#     pickle.dump({"sentences": sentences, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Store sentences & embeddings on disc
import pickle
with open(input_dir+"/embeddings.pkl", "wb") as fOut:
    pickle.dump({"sentences": sentences, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

import pickle

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...
#



# Load sentences & embeddings from disc
import torch
with open(input_dir+"/embeddings.pkl", "rb") as f:
    #map_location=torch.device('cpu')
    #stored_data=torch.load(fIn,map_location=torch.device('cpu'))
    stored_data  = CPU_Unpickler(f).load()
    #stored_data = pickle.load(fIn,map_location='cpu', pickle_module=pickle)
    stored_sentences = stored_data["sentences"]
    stored_embeddings = stored_data["embeddings"]



# sentences1,labels1,embedding1=[],[],[]

# classes=['Algeria','Tunisia','Egypt','Sudan','Jordan','Palestine','Sudan','Syria','Tunisia','Yemen']
# for i,sent in enumerate(sentences):
#   if labels[i] in classes:
#     sentences1.append(sent)
#     labels1.append(labels[i])
#     embedding1.append(stored_embeddings[i])

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

# # This function will search all wikipedia articles for passages that
# # answer the query
# from collections import Counter
# bm25_results=[]
# cross_enc_results=[]
# enc_results=[]
# def search(query):
#     print("Input question:", query)

#     ##### BM25 search (lexical search) #####
#     #bm25_scores = bm25.get_scores(bm25_tokenizer(query))
#     bm25_scores = bm25.get_scores(query)
#     top_n = np.argpartition(bm25_scores, -100)[-100:]
#     print("No. of items retrieved:",len(top_n))
#     bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
#     bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
#     print("bm25_hits:",bm25_hits)
#     #df_bm25 = pd.DataFrame(bm25_hits)
#     #bm25_results.append(bm25_hits)
#     print("Top-3 lexical search (BM25) hits")
#     c=[]
#     for hit in bm25_hits[0:3]:
#         print("\t{:.3f}\t{}\t{}\t{}".format(hit['score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]))
#     for hit in bm25_hits:
#       c.append([query,hit['score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]])



#     # labs=[x[-1] for x in c]
#     # sc=[x[1] for x in c]
#     # l=Counter(labs)
#     # l=l.most_common()
#     # #print(l)
#     # c1,c2,c3=[],[],[]
#     # for i in c:
#     #     c1.append(i + [l])
#     # for i in c1:
#     #     c2.append(i + [sc])
#     # for i in c2:
#     #     c3.append(i + [labs])
#     #     # c1.append(i+[sc])
#     #     # c1.append(i+[labs])
#     # #c1=[l1.append(l) for l1 in c]
#     # #print(c1)
#     # bm25_results.append(c3)
#     # print(len(bm25_results))



#     ##### Re-Ranking #####
#     # Now, score all retrieved passages with the cross_encoder
#     cross_inp = [[query, sentences[hit['corpus_id']]] for hit in bm25_hits]
#     cross_scores = cross_encoder.predict(cross_inp)

#     # Sort results by the cross-encoder scores
#     for idx in range(len(cross_scores)):
#         bm25_hits[idx]['cross-score'] = cross_scores[idx]

#     print("\n-------------------------\n")
#     print("Top-3 Cross-Encoder Re-ranker hits")
#     hits = sorted(bm25_hits, key=lambda x: x['cross-score'], reverse=True)
#     #cross_enc_results.append(hits)
#     for hit in hits[0:3]:
#         print("\t{:.3f}\t{}\t{}\t{}".format(hit['cross-score'],sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]))
#     e=[]
#     #Get the re-ranked top 10
#     for hit in hits:
#       e.append([query,hit['cross-score'], sentences[hit['corpus_id']].replace("\n", " "),hit['corpus_id'],labels[hit['corpus_id']]])

#     labs=[x[-1] for x in e]
#     sc=[x[1] for x in e]
#     l=Counter(labs)
#     l=l.most_common()
#     #print(l)
#     e1,e2,e3=[],[],[]
#     for i in e:
#         e1.append(i + [l])
#     for i in e1:
#         e2.append(i + [sc])
#     for i in e2:
#         e3.append(i + [labs])

#     cross_enc_results.append(e3)
#     print(len(cross_enc_results))

#results = co.rerank(query=query, documents=, top_n=3, model="rerank-multilingual-v2.0")



path1=i"./NADI2024_subtask1_dev2.tsv"



import pandas as pd
path_test=path1
dev=pd.read_csv(path_test,delimiter="\t")

#labels=data.iloc[:,1:]

queries=dev['sentence'].values

#queries=[removeNonArabicChar(x) for x in queries]

import pandas as pd
path_test=path1
test=pd.read_csv(path_test,delimiter="\t")
queries=test['sentence'].values



#'''Algeria,Egypt,Jordan,Palestine,Sudan,Syria,Tunisia,Yemen

#apply_softmax=True
#from torch import nn
#model = CrossEncoder(model_name, default_activation_function=nn.Sigmoid())

from collections import Counter
for q in queries:

  search(query = q)
  print("################\n")

data = [v for lst in cross_enc_results for v in lst]

#data = [v for lst in bm25_results for v in lst]
#bm25+cross_enc
out_path="./NADI_2024/"

df_enc_cross = pd.DataFrame(data, columns=['Query','Score','Sentence','Sent_Id','Label','Label_Counter','Score_List','Label_list'])
df_enc_cross_unique=df_enc_cross.drop_duplicates(subset='Query')
df_enc_cross.to_csv(out_path+"/df_enc_cross_Test.csv")
preds_cross=df_enc_cross_unique['Label_Counter'].values



import ast
label_lists=df_enc_cross_unique['Label_list'].values
label_lists=[ast.literal_eval(x) for x in label_lists]
label_lists=[list(set(x)) for x in label_lists]


import ast
preds_cross=[ast.literal_eval(x) for x in preds_cross]

"""**APPROACH 1: COUNT BASED THRESHOLDS**"""

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

# preds1=final_preds(preds_cross)

preds_cross1=pred_counts(preds_cross)
preds1=final_preds(preds_cross1)
df_preds=pd.DataFrame(preds1)
df_preds = df_preds[COUNTRIES]

df_preds = df_preds.replace(np.nan, 0)
df_preds = df_preds.replace('y', 1)
df_preds = df_preds.replace('n', 0)
pred_labels=df_preds.values.tolist()
pred_labels1=[','.join((map(str,s))) for s in pred_labels]
'''
Unsupervised Cross-Encoder with Count based Thresholds (Un-Cross-LCT)
'''
with open (out_path+"/Results/NLP_DI_subtask1_cross_Test","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")


"POST-FILTERING"

if filtering:

    def select(n,p):
      p1=p[:n]
      return p1
    
    multi_preds_binarized=[]
    multi_preds_binarized2=[]
    '''
    N=8
    '''
    for k in list(range(8,9)):
      multi_preds=[]
      mm=[]
      for preds in label_lists_idx:
        #print(k)
        s=select(k,preds)
        #print(s)
        m = [1 if i in s else 0 for i in list(range(0,18))]
        #m1=[x if i in dev_labels_map else 0 for i,x in enumerate(m)]# considering only dev labels
        mm.append(m)
        #multi_preds.append(m1)
      #multi_preds_binarized.append(multi_preds)
        multi_preds_binarized2.append(m)
    with open (out_path+"/Results/NLP_DI_subtask1_cross_8","w") as f:
      for line in multi_preds_binarized2:
        line = ",".join(str(element) for element in line)
        #line="".join(str(line))
        f.write(f"{line}\n")

# multi_preds_binarized=[]
# multi_preds_binarized2=[]
# for k in list(range(4,5)):
#   multi_preds=[]
#   mm=[]
#   for preds in label_lists_idx:
#     #print(k)
#     s=select(k,preds)
#     #print(s)
#     m = [1 if i in s else 0 for i in list(range(0,18))]
#     #m1=[x if i in dev_labels_map else 0 for i,x in enumerate(m)]# considering only dev labels
#     mm.append(m)
#     #multi_preds.append(m1)
#   #multi_preds_binarized.append(multi_preds)
#     multi_preds_binarized2.append(m)
    


"LABEL ENHANCEMENTS"
maps= {'Algeria': 0, 'Bahrain': 1, 'Egypt': 2, 'Iraq': 3, 'Jordan': 4, 'Kuwait': 5, 'Lebanon': 6, 'Libya': 7, 'Morocco': 8, 'Oman': 9, 'Palestine': 10, 'Qatar': 11, 'Saudi_Arabia': 12, 'Sudan': 13, 'Syria': 14, 'Tunisia': 15, 'UAE': 16, 'Yemen': 17}

rev_maps={v:k for k,v in maps.items()}


"Co-occurrence Based+ Post Filterings"


import pandas as pd
df_test_corr = pd.DataFrame(preds, columns = COUNTRIES)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import colors as c
#cMap = c.ListedColormap(['y','b'])
corr = df_test_corr.fillna(0).corr()
print(corr)
plt.pcolormesh(corr)
plt.show()

out_test_corr = corr.gt(0.0).apply(lambda x: x.index[x].tolist(), axis=1)
t1=out_test_corr.to_dict()

test_t1={'Algeria': ['Algeria', 'Libya', 'Morocco', 'Tunisia'],
 'Bahrain': ['Bahrain',
  'Iraq',
  'Kuwait',
  'Oman',
  'Qatar',
  'Saudi_Arabia',
  'UAE'],
 'Egypt': ['Egypt', 'Palestine'],
 'Iraq': ['Bahrain', 'Iraq', 'Morocco', 'Yemen'],
 'Jordan': ['Jordan', 'Lebanon', 'Palestine', 'Syria'],
 'Kuwait': ['Bahrain', 'Kuwait', 'Oman', 'Qatar', 'Saudi_Arabia', 'UAE'],
 'Lebanon': ['Jordan', 'Lebanon', 'Palestine', 'Syria'],
 'Libya': ['Algeria', 'Libya', 'Tunisia'],
 'Morocco': ['Algeria', 'Iraq', 'Morocco', 'Tunisia'],
 'Oman': ['Bahrain',
  'Kuwait',
  'Oman',
  'Qatar',
  'Saudi_Arabia',
  'UAE',
  'Yemen'],
 'Palestine': ['Egypt', 'Jordan', 'Lebanon', 'Palestine', 'Syria', 'Yemen'],
 'Qatar': ['Bahrain',
  'Kuwait',
  'Oman',
  'Qatar',
  'Saudi_Arabia',
  'UAE',
  'Yemen'],
 'Saudi_Arabia': ['Bahrain',
  'Kuwait',
  'Oman',
  'Qatar',
  'Saudi_Arabia',
  'UAE',
  'Yemen'],
 'Sudan': ['Sudan'],
 'Syria': ['Jordan', 'Lebanon', 'Palestine', 'Syria'],
 'Tunisia': ['Algeria', 'Libya', 'Morocco', 'Tunisia'],
 'UAE': ['Bahrain', 'Kuwait', 'Oman', 'Qatar', 'Saudi_Arabia', 'UAE'],
 'Yemen': ['Iraq', 'Oman', 'Palestine', 'Qatar', 'Saudi_Arabia', 'Yemen']}

custom_t1={'Algeria': ['Algeria', 'Sudan', 'Tunisia'],
 'Egypt': ['Egypt', 'Palestine', 'Sudan', 'Tunisia'],
 'Jordan': ['Jordan', 'Palestine', 'Sudan', 'Syria', 'Yemen'],
 'Palestine': ['Egypt', 'Jordan', 'Palestine', 'Sudan', 'Syria', 'Yemen'],
 'Sudan': ['Algeria',
  'Egypt',
  'Jordan',
  'Palestine',
  'Sudan',
  'Syria',
  'Tunisia',
  'Yemen'],
 'Syria': ['Jordan', 'Palestine', 'Sudan', 'Syria', 'Tunisia', 'Yemen'],
 'Tunisia': ['Algeria', 'Egypt', 'Sudan', 'Syria', 'Tunisia'],
 'Yemen': ['Jordan', 'Palestine', 'Sudan', 'Syria', 'Yemen'],
  'Saudi_Arabia': ['Saudi_Arabia','Yemen'],'Lebanon': ['Jordan', 'Lebanon', 'Palestine', 'Syria']}

#Egyptian,  Algerian, Syrian, Palestinian, Jordan, Sudan, Tunisia, Yemen
 #Saudi Arabian,Lebanese

dev_t1={'Algeria': ['Algeria', 'Sudan', 'Tunisia'],
 'Egypt': ['Egypt', 'Palestine', 'Sudan', 'Tunisia'],
 'Jordan': ['Jordan', 'Palestine', 'Sudan', 'Syria', 'Yemen'],
 'Palestine': ['Egypt', 'Jordan', 'Palestine', 'Sudan', 'Syria', 'Yemen'],
 'Sudan': ['Algeria',
  'Egypt',
  'Jordan',
  'Palestine',
  'Sudan',
  'Syria',
  'Tunisia',
  'Yemen'],
 'Syria': ['Jordan', 'Palestine', 'Sudan', 'Syria', 'Tunisia', 'Yemen'],
 'Tunisia': ['Algeria', 'Egypt', 'Sudan', 'Syria', 'Tunisia'],
 'Yemen': ['Jordan', 'Palestine', 'Sudan', 'Syria', 'Yemen']}


out_path=input_dir+"/NADI_2024/"
import ast

with open (out_path+"/Results/NLP_DI_subtask1_cross_with_corr_4","w") as f:
  for line in multi_preds_binarized2:
    line = ",".join(str(element) for element in line)
    #line="".join(str(line))
    f.write(f"{line}\n")

label_lists_idx=[[maps[x] for x in l] for l in label_lists]

label_lists_max=[x[0] for x in label_lists]


preds_cross_max = [item[0] for item in preds_cross]
preds_cross_max[:2]

import itertools
label_lists1=[]
for label in label_lists:
  ext_labels=[dev_t1[x] if x in dev_t1.keys() else [x] for x in label ]
  ext_labels=list(itertools.chain(*ext_labels))
  print(ext_labels)
  ext_labels=list(set(ext_labels))
  label_lists1.append(ext_labels)


label_lists_idx=[[maps[x] for x in l] for l in label_lists1]


import itertools
label_lists1=[]
for label in label_lists:
  ext_labels=[custom_t1[x] if x in custom_t1.keys() else [x] for x in label ]
  ext_labels=list(itertools.chain(*ext_labels))
  print(ext_labels)
  ext_labels=list(set(ext_labels))
  label_lists1.append(ext_labels)
label_lists_idx=[[maps[x] for x in l] for l in label_lists1]
multi_preds_binarized=[]
multi_preds_binarized2=[]
for k in list(range(8,9)):
  multi_preds=[]
  mm=[]
  for preds in label_lists_idx:
    #print(k)
    s=select(k,preds)
    #print(s)
    m = [1 if i in s else 0 for i in list(range(0,18))]
    #m1=[x if i in dev_labels_map else 0 for i,x in enumerate(m)]# considering only dev labels
    mm.append(m)
    #multi_preds.append(m1)
  #multi_preds_binarized.append(multi_preds)
    multi_preds_binarized2.append(m)

with open (out_path+"/Results/NLP_DI_subtask1_cross_with_custom_corr_8","w") as f:
  for line in multi_preds_binarized2:
    line = ",".join(str(element) for element in line)
    #line="".join(str(line))
    f.write(f"{line}\n")

import itertools
label_lists1=[]
for label in label_lists:
  ext_labels=[test_t1[x] if x in test_t1.keys() else [x] for x in label ]
  ext_labels=list(itertools.chain(*ext_labels))
  print(ext_labels)
  ext_labels=list(set(ext_labels))
  label_lists1.append(ext_labels)
label_lists_idx=[[maps[x] for x in l] for l in label_lists1]
multi_preds_binarized=[]
multi_preds_binarized2=[]
for k in list(range(8,9)):
  multi_preds=[]
  mm=[]
  for preds in label_lists_idx:
    #print(k)
    s=select(k,preds)
    #print(s)
    m = [1 if i in s else 0 for i in list(range(0,18))]
    #m1=[x if i in dev_labels_map else 0 for i,x in enumerate(m)]# considering only dev labels
    mm.append(m)
    #multi_preds.append(m1)
  #multi_preds_binarized.append(multi_preds)
    multi_preds_binarized2.append(m)

with open (out_path+"/Results/NLP_DI_subtask1_cross_with_test_corr_8","w") as f:
  for line in multi_preds_binarized2:
    line = ",".join(str(element) for element in line)
    #line="".join(str(line))
    f.write(f"{line}\n")

"Confusion MAtrix based"



#First we get the mono-label predictions from the fine tuned NADI 2023 model, which is stored in the file NLP_DI_subtask1_dev_18

with open(out_path+"NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_18","r") as f: #MONOLABEL_BASELINE (MULTI-CLASS PREDICTIONS)
  f=f.readlines()
  preds_mono=[ast.literal_eval(x) for x in f]
indices = [[i for i, x in enumerate(p) if x == 1][0] for p in preds_mono]



multi_map={0:{3:2,4:1,8:1,10:1,14:1,15:5},1:{3:1,4:1,5:11,7:1,9:2,11:4,12:1,16:6},
           2:{7:2,10:2,12:1,16:1},3:{9:1,10:1,12:1,16:1,17:1},4:{1:2,3:1,5:2,6:1,9:1,10:12,14:2,16:1},
           5:{1:15,3:1,11:2,12:1,16:1},6:{5:1,10:4,14:11,16:1},7:{0:2,2:1,4:1,10:1,11:1,14:2,15:2,16:1,17:1},
           8:{0:7,10:1,15:3,17:1},9:{1:2,3:1,5:1,6:1,12:4,16:5},10:{2:2,3:1,4:9,5:1,6:2,9:2,14:1,15:2,16:1},
           11:{1:8,3:2,4:1,5:6,9:1,12:3,16:6,17:1},12:{1:2,3:1,5:5,9:3,11:7,16:1,17:6},
           13:{2:1,4:1,7:1,10:1,14:1,17:1},14:{4:8,6:6,10:1},15:{0:4,7:4},16:{1:5,4:1,5:4,8:1,9:6,10:1,11:4,12:4},
           17:{1:1,12:2,14:1,16:1}}

#preds_idx=[[i for i,x in enumerate(p) if x==1] for p in preds]



multi_=[list(multi_map[x].keys()) for x in multi_map]


#Now enhance the labels
multi_preds=[]
multi_preds_binarized=[]
for pred_key in indices:

  multilabels_possible = multi_map[pred_key]
  multi_=list(multilabels_possible.keys())
  multi_=[x for x in list(multilabels_possible.keys())]
  multi_.extend([pred_key])
  #multi_=[x for x in multi_ if x in dev_labels_map]
  m = [1 if i in multi_ else 0 for i in list(range(0,18))]
  multi_preds.append(multi_)
  multi_preds_binarized.append(m)

with open (out_path+"/NLP_DI_subtask1_dev_CMbasedMONO","w") as f:
  for line in multi_preds_binarized:
    line = ",".join(str(element) for element in line)
    #line="".join(str(line))
    f.write(f"{line}\n")



# final_preds=[]
# final_preds1=[]
# import ast

# #based on corr

# #p=[x[0] if maps[x[0]] in dev_labels_map else 0 for x in preds_cross_max ]
# p=[x[0] for x in preds_cross_max ]
# for k in p:
#   if k==0:
#     p1=[0] * 18
#   else:
#     t1=t[k]
#     p1=[1 if k in t1 else 0 for k in COUNTRIES]

#   final_preds.append(p1)




# multi_preds_binarized=[]
# multi_preds_binarized2=[]
# for k in list(range(8,9)):
#   multi_preds=[]
#   mm=[]
#   for preds in label_lists_idx:
#     #print(k)
#     s=select(k,preds)
#     #print(s)
#     m = [1 if i in s else 0 for i in list(range(0,18))]
#     #m1=[x if i in dev_labels_map else 0 for i,x in enumerate(m)]# considering only dev labels
#     mm.append(m)
#     #multi_preds.append(m1)
#   #multi_preds_binarized.append(multi_preds)
#     multi_preds_binarized2.append(m)



# with open (out_path+"/Results/NLP_DI_subtask1_cross_8","w") as f:
#   for line in multi_preds_binarized2:
#     line = ",".join(str(element) for element in line)
#     #line="".join(str(line))
#     f.write(f"{line}\n")


    

'''

"""**DEV EXPERIMENTS**"""

df_bm = pd.DataFrame(data, columns=['Query','Score','Sentence','Sent_Id','Label','Label_Counter','Score_List','Label_list'])

df_bm.head()

#df_bm_unique=df_bm.groupby('Query').first().reset_index()

df_bm_unique=df_bm.drop_duplicates(subset='Query')

data = [v for lst in enc_results for v in lst]
df_enc = pd.DataFrame(data, columns=['Query','Score','Sentence','Sent_Id','Label','Label_Counter','Score_List','Label_list'])

df_enc.head()

#df_enc_unique=df_enc.groupby('Query').first().reset_index()

df_enc_unique=df_enc.drop_duplicates(subset='Query')

len(cross_enc_results)

cross_enc_results[:2]

data = [v for lst in cross_enc_results for v in lst]

df_enc_cross = pd.DataFrame(data, columns=['Query','Score','Sentence','Sent_Id','Label','Label_Counter','Score_List','Label_list'])

df_enc_cross.head()

#df_enc_cross_unique=df_enc_cross.groupby('Query').first().reset_index()

df_enc_cross_unique=df_enc_cross.drop_duplicates(subset='Query')

df_enc_cross_unique

path

input_dir

input_dir

out_path=input_dir+"/NADI_2024/"

df_bm.to_csv(out_path+"/df_bm2.csv")
df_enc.to_csv(out_path+"/df_enc2.csv")
df_enc_cross.to_csv(out_path+"/df_enc_cross2.csv")

df_bm_unique.to_csv(out_path+"/df_bm_unique2.csv")
df_enc_unique.to_csv(out_path+"/df_enc_unique2.csv")
df_enc_cross_unique.to_csv(out_path+"/df_enc_cross_unique2.csv")

df_bm.to_csv(out_path+"/df_bm3.csv")# only with classes in dev2 set
df_enc.to_csv(out_path+"/df_enc3.csv")
df_enc_cross.to_csv(out_path+"/df_enc_cross3.csv")
df_bm_unique.to_csv(out_path+"/df_bm_unique3.csv")
df_enc_unique.to_csv(out_path+"/df_enc_unique3.csv")
df_enc_cross_unique.to_csv(out_path+"/df_enc_cross_unique3.csv")

import pandas as pd
df_bm_unique=pd.read_csv(out_path+"/df_bm_unique2.csv")
df_enc_unique=pd.read_csv(out_path+"/df_enc_unique2.csv")
df_enc_cross_unique=pd.read_csv(out_path+"/df_enc_cross_unique2.csv")

df_enc_cross_unique.head()

df_enc_unique[:20]

df_bm_unique.head()

true_labels=dev.iloc[:,1:]

import numpy as np
true_labels = true_labels.replace(np.nan, 0)
true_labels = true_labels.replace('y', 1)
true_labels = true_labels.replace('n', 0)

true_labels=true_labels.values.tolist()

len(true_labels)

true_labels[:6]

#Algeria,Egypt,Sudan,Tunisia

#Algeria,Egypt,Jordan,Palestine,Sudan,Syria,Tunisia,Yemen

import ast
preds_bm=df_bm_unique['Label_Counter'].values
preds_enc=df_enc_unique['Label_Counter'].values
preds_cross=df_enc_cross_unique['Label_Counter'].values
#preds_bm=[ast.literal_eval(x) for x in preds_bm]
preds_bm[:5]

preds_enc[:5]

preds_cross[:5]

def pred_counts(pred):
  preds=[]
  for i,item in enumerate(pred):
    p=[]
    j=0
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

"""if the no. of labels predicted in top 10 <=3"""

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

preds=pred_counts(preds_bm)



len(preds)

preds[:15]

class_names=list(set(labels))
class_names

len(class_names)

# preds1=[]
# l=class_names
# for q in preds:
#   p1={}
#   #for q in p:
#   c1=[x[0] for x in q]
#   #print(c1)
#   c2=[x[1] for x in q]
#   if len(c1)!=len(l):
#     added = list(sorted(set(l) - set(c1)))
#     print(added)

#   for i,y in enumerate(c1):
#     if y=='Algeria':
#       p1['Algeria']=c2[i]
#     elif y=='Egypt':
#       p1['Egypt']=c2[i]
#     elif y=='Sudan':
#       p1['Sudan']=c2[i]
#     elif y=='Tunisia':
#       p1['Tunisia']=c2[i]
#   if added[0] not in list(p1.keys()):
#     p1[added[0]]='n'
#   preds1.append(p1)

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

preds[:5]

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

preds1=final_preds(preds)

preds1[0]

df_preds=pd.DataFrame(preds1)
df_preds = df_preds[COUNTRIES]

df_preds = df_preds.replace(np.nan, 0)
df_preds = df_preds.replace('y', 1)
df_preds = df_preds.replace('n', 0)
pred_labels=df_preds.values.tolist()
pred_labels1=[','.join((map(str,s))) for s in pred_labels]

df_preds.head()

pred_labels[:2]

assert len(pred_labels)==len(true_labels)

out_path

with open(out_path+"/NADI2024-Dev2_data/subtask1/NADI2024_subtask1_dev2_gold.txt","r") as f:
  f1=f.readlines()

f1[0]

with open(out_path+"/NADI2024-Dev2_data/subtask1/UBC_subtask1_dev_1.txt","r") as f:
  f2=f.readlines()

f2[0]

!ls

input_dir

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/gdrive/My Drive/Colab Notebooks/INCdid_Project/NADI/NADI_2024/"

!ls

! python ./NADI2024-Dev2_data/subtask1/NADI2024-ST1-Scorer.py ./NADI2024-Dev2_data/subtask1/NADI2024_subtask1_dev2_gold.txt ./NADI2024-Dev2_data/subtask1/UBC_subtask1_dev_1.txt

pred_labels1[:2]

true_labels[:5]

'''
#Algeria,Egypt,Jordan,Palestine,Sudan,Syria,Tunisia,Yemen
'''

print(COUNTRIES)

preds_bm[:5]

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_1","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")



with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_6","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_9","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

preds_enc[:5]

preds_cross[:5]

preds_enc1=pred_counts(preds_enc)

preds1=final_preds(preds_enc1)
df_preds=pd.DataFrame(preds1)
df_preds = df_preds[COUNTRIES]

df_preds = df_preds.replace(np.nan, 0)
df_preds = df_preds.replace('y', 1)
df_preds = df_preds.replace('n', 0)
pred_labels=df_preds.values.tolist()
pred_labels1=[','.join((map(str,s))) for s in pred_labels]

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_2","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_7","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_10","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

preds_cross1=pred_counts(preds_cross)

preds_cross1=pred_counts(preds_cross)
preds1=final_preds(preds_cross1)
df_preds=pd.DataFrame(preds1)
df_preds = df_preds[COUNTRIES]

df_preds = df_preds.replace(np.nan, 0)
df_preds = df_preds.replace('y', 1)
df_preds = df_preds.replace('n', 0)
pred_labels=df_preds.values.tolist()
pred_labels1=[','.join((map(str,s))) for s in pred_labels]

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_3","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_8","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_11","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_bm25cross","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

with open (out_path+"/NADI2024-Dev2_data/subtask1/NLP_DI_subtask1_dev_bm25cross_dev","w") as f:
  for line in pred_labels1:
    line="".join(str(line))
    f.write(f"{line}\n")

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

print (COUNTRIES)

'''

