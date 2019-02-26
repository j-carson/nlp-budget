
# coding: utf-8

# # Common code to read and process the corpus
# 
# To make sure all models are preprocessing the data in a similar way

# In[24]:


import pandas as pd
import numpy as np 
import pdb
import json
import re
import pickle

from pathlib import Path


# In[25]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer


# In[26]:


docdir = Path('../data/docs')
datadir = Path('../data')


# In[27]:


def read_raw_corpus():
    raw_corpus = []

    files = list(docdir.glob('*.body'))
    header_files = [ (str(f)).replace('.body', '.heading') for f in files ]
    
    for body, header in zip(files, header_files):
        with open(header, 'r') as fp:
            head_dict = json.load(fp)
            fp.close()
            
        doc = ''
        for key in ['division', 'title', 'major', 'inter', 'small']:
            if head_dict[key] != '':
                doc += head_dict[key] + '\n'
        
        with open(body, 'r') as fp:
            doc = '\n'.join([doc, fp.read()])
            fp.close()
        raw_corpus.append(doc)

    return raw_corpus


# In[28]:


def tokenize_raw_budget(raw_corpus):
    corpus = []
    
    # tla = three (or more) all-capitialized acronymns
    tla = re.compile(r'[A-Z]{3,}\S*?\b')
    p = PorterStemmer()
    
    for doc in raw_corpus:
        # some headings are in all caps - don't mix those up with acronyms
        headings, body =  doc.split('\n\n')
        body = headings + ' ' + re.sub(tla, 'tla', body)
        preproc = simple_preprocess(body, min_len=3)
        stemmed = [ p.stem(w) for w in preproc ]
        corpus.append(stemmed)
        
    return corpus


# In[29]:


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


