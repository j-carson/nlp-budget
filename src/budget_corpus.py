# coding: utf-8

# # Common code to read and process the corpus
# 
# To make sure all models are preprocessing the data in a similar way


import pandas as pd
import numpy as np 
import pdb
import json
import re
import pickle

from pathlib import Path

import gensim
from gensim.utils import simple_preprocess
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS as gs_stopwords

docdir = Path('../data/docs')
datadir = Path('../data')


def read_raw_corpus():
    raw_corpus = []

    files = list(docdir.glob('*.body'))
    
    for body in files:
        with open(body, 'r') as fp:
            doc = fp.read()
            fp.close()
        raw_corpus.append(doc)

    return raw_corpus


def tokenize_raw_budget(raw_corpus):
    
    # -- stemming would reduce some duplication here (e.g. section and sections)
    # -- but that results in a lot of non-words that don't work with word2vec
    # -- (Actually the lemmatizer may be fixing some of these, could go back
    # -- and check. simple_precprocess was leaving the plurals when I wrote this
    # -- list.)
    
    stopwords = set( """section sections subsection subsections chapter part amended
        state federal local district grant amount amounts  
        fund funding cost expense expend expenditure purchase account fiscal 
        pay payment payments
        provide provision proviso
        administration administrative administrator
        budget purposes united states national general government
        office regulations act acts title code 
        specified provided available further including herein
        enactment program service operation operations activity
        appropriation appropriated provision provisions department agency
        those
        aaa bbb ccc ddd eee""".split() )
    stopwords.update(gs_stopwords)
    
    # tla = three (or more) letter all-capitialized acronymns
    tla = re.compile(r'[A-Z]{3,}\S*?\b')
    
    corpus = []
    for doc in raw_corpus:
        body = re.sub(tla, '', doc)
        # from this lemmatization tutorial
        # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
        lemmatized = [wd.decode('utf-8').split('/')[0] for wd in lemmatize(body,  min_length=3)]
        tokens = [ t for t in lemmatized if t not in stopwords ]
        corpus.append(tokens)
        
    return corpus


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

