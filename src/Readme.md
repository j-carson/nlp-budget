# About the files in this directory 

| File | About | 
|:--|:---|
| corpus.ipynb | %loads the budget_corpus.py file for testing |
| budget_corpus.py | Included in all the analysis notebooks so the documents are preprocessed the same in all tests |
| unique_words.ipynb | Demonstrates the preponderance of low frequency words in the corpus.
| clustering.ipynb | An attempt at clustering. This one does LSA to reduce dimensionality followed by HDBSCAN to cluster. HDBSCAN was chosen as my clusterer-of-choice because I knew that there would be unclassifiable documents, I just didn't anticipate the extent of the problem. | 
| lda_cluster.ipynb | Another attempt at clustering. This one was to play with the LDA function to get experience with the API. This corpus is not suited for LDA analysis. |
| word2vec.ipynb | Using Word Mover Distance to try to cluster the documents. Word Mover Distance uses a soft-similarity metric to classify based on an underlying word vector model rather than requiring the clustered documents to have words in commmon. Due to how slow this is to run, I chose a subset of shorter documents. |
| smart_stopwords.py | This is the list of stop words used to test the Word Mover distance. It was the one used in the WMD paper referenced on the gensim site. Since Word Mover uses skip-grams, I did not filter as many words as I did with the other tests to keep the words in their contexts. | 
| glove.ipynb | This was another attempt to work with word vectors. In this case, I used glove to see if my low-frequency words could be organized into clusters. Again, I clustered with HDBSCAN. In this case, I found some word clusters that could be used to data mine the budget, but definitely mixed with a lot of noise. |
