
# coding: utf-8

# # Common code to read and process the corpus
# 
# To make sure all models are preprocessing the data in a similar way

# In[97]:


import pandas as pd
import numpy as np 
import pdb
import json
import re
import pickle

from pathlib import Path


# In[98]:


import gensim
from gensim.utils import simple_preprocess
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS as gs_stopwords


# In[99]:


docdir = Path('../data/docs')
datadir = Path('../data')


# In[108]:


def read_raw_corpus():
    raw_corpus = []

    files = list(docdir.glob('*.body'))
    
    for body in files:
        with open(body, 'r') as fp:
            doc = fp.read()
            fp.close()
        raw_corpus.append(doc)

    return raw_corpus


# In[109]:


def tokenize_raw_budget(raw_corpus):
    corpus = []
    stopwords =         """for the this that which such than with within without after from 
        and but use each more less unless any law carry out 
        has have are will shall may were been who its into subsection amount state grants 
        fund funds funded costs expenses expended expenditure purchase account administration administrative
        budget purposes during  united states national general government
        house senate congress president office regulations act title code 
        specified provided available further including herein
        enactment program programs services operation operations activities activity
        appropriation appropriations appropriated privision provisions agency agencies""".split()
    
    
    # tla = three (or more) letter all-capitialized acronymns
    tla = re.compile(r'[A-Z]{3,}\S*?\b')
    
    for doc in raw_corpus:
        body = re.sub(tla, '', doc)
        tokens = simple_preprocess(body, min_len=3)
        tokens = [ t for t in tokens if t not in stopwords ]
        corpus.append(tokens)
        
    return corpus


# In[110]:


def read_documents():
    doc_pickle = datadir / 'corpus.pkl'
    if doc_pickle.exists():
        with open(doc_pickle, 'rb') as fp:
            corpus = pickle.load(fp)
    else:
        raw_corpus = read_raw_corpus()
        corpus = tokenize_raw_budget(raw_corpus)
        # put it in a pickle file
        with open(doc_pickle, 'wb') as fp:
            pickle.dump(corpus, fp)
    return corpus

