# Project Summary

Using natural language processing to examine the federal budget

# Background

The idea for this project came from an [article I saw](https://www.texastribune.org/2019/02/14/government-shutdown-deal-includes-protections-south-texas-landmarks/) which mentioned that the most recent budget bill attempts to spare the 
National Butterfly Reserve from being destroyed by border wall construction projects. 

From the perspective of someone studying natural language processing, I wondered 
- How often does a word like 'butterfly' appear in the Federal budget? 
- Could I use natural language processing to find provisions added to long, complex pieces of legislation to address particular local issues or special interests? 

# Need for study

Slipping extra provisions into 'must pass' bills is a common legislative tactic. A machine learning model that can identify 
unusual language or provisions in a bill could be useful to multiple audiences:

- Congressional staffers
- Political activists
- Lobbyists
- Journalists
- Concerned citizens

# Data

The dataset is H.J. Res 31 (Enrolled Bill), the bill which ended the recent shutdown
and funds the government through September, 2019. The PDF vesion of this bill is 465 pages long. 
I worked with the [XML file](https://www.congress.gov/116/bills/hjres31/BILLS-116hjres31enr.xml), which I parsed using BeautifulSoup.

# Tools

The following tools were used:

  - Python
  - Gensim : A package focused on document similarity
  - BeautifulSoup : for processing XML
  - Word2Vec and Glove : attempting to group words by shared meanings

# Preprocessing

The use of XML tags through the various sections of the budget was not consistent, but I was able to use heuristics to collect the blobs of text between headings and subheadings, as well as the subheadings themselves. This gave me a dataset of 1248 documents.

The documents are very uneven in size. For example, see any of the small allocations on page 3 or 4 [PDF link](https://www.congress.gov/116/bills/hjres31/BILLS-116hjres31enr.pdf), or the long list of conditions placed on State Department funding starting on page 312.


# Exploratory Data Analysis

My initial attempts at exploratory data analysis were useful in uncovering parsing errors. Once those were out of the way, I was able to categorize a typical document as:

- __Numbers__. A budget has a lot of these.
- __Stop words__: what we think of as typical English stop words like 'the' or 'and'
- __Budget/legal stopwords__.  These are terms like cost, expense, expenditure, account, 
title, sections, subsection, law. These words that indicate that I'm in a financial/government
document, but they weren't terms that I wanted to use to group documents by topic. 
- __Unique words__. Most of the remaining words are either unqiue to the particular section of the
budget we are analyzing, or are only repeated in a small number of other sections.

## It turns out are a lot of unique words in the budget

Notebook ```unique_words.ipynb```

After parsing the data into words and removing numbers and stop words, there were 1600 words (out of a 4000-ish word vocabulary) which were
used in exactly one document. The majority of the vocabulary (after removing stopwords) is used in 5 percent or fewer of the documents.

There is no single explanation for the unique words. 
Sometimes it's because there's an unusual proviso, like the proviso for butterfly habitat. Sometimes it's just a side effect of 
the fact that the document is a list of all the departments of the government. For example, the Supreme Court budget is the only section with
the word 'supreme.' Some of the unique words are 
common English words like 'beginning.' These are just words that you might expect to see more
than once in a 200,000+ word sample of English, but the federal budget is just not a 
normal sample of English writing.


# Results

## Cluster analysis

Source code: ``clustering.ipynb``

Due to the prevalence of unique words, the Federal budget is hard to cluster or partition in an unsupervised way. The underlying assumption with any approach to clustering is that the data is not scattered randomly across all its dimensions. Unfortunately,
that's basically the case for this corpus.

Traditional cluster analysis did allow me to find
sections of the budget that are exact duplicates or nearly so. For example, this text (with a different section number) appears in seven locations: 

>SEC.  523.  (a)  None  of  the  funds  made  available  in  this  Act  may  be  used  to  maintain  or  establish  a  computer  network  unless  such  network  blocks  the  viewing,  downloading,  and  exchanging  of pornography. (b)  Nothing  in  subsection  (a)  shall  limit  the  use  of  funds  necessary for any Federal, State, tribal, or local law enforcement agency or any other entity carrying out criminal investigations, prosecution, or adjudication activities. 

## Using word vectors to create clusters


Using word vectors such as Word2vec and Glove, we can 
group words that appear in similar contexts, even when they are not 
identical words. This can be done at the level of individual words 
or at the document level. 

Additional preprocessing was done for both of the following tests to 
remove words from the corpus that were not in the pretrained models. 

### Clustering individual budget words 

Source code: ``glove.ipynb``

I clustered individual budget words based on their proximity in the pretrained Glove
model we downloaded in class.

I found some clusters that made logical sense:

__Energy policy__
'supply', 'energy', 'gas', 'oil', 'mining', 'fuel', 'coal', 'crude', 'petroleum', 'demand', 'electricity'

__Latin America__
'mexico', 'colombia', 'guatemala', 'honduras', 'venezuela', 'nicaragua'

__Public Health__
'prevention', 'avian', 'bird', 'contagious', 'disease', 'infectious', 'outbreak', 'aids', 'suffer', 'ebola', 'virus', 'vaccine', 'pathogen', 'immunization', 'malaria', 'polio', 'tuberculosis', 'severe', 'illness', 'epidemic', 'bacterial', 'infection', 'vaccination'

And some that were less helpful:

__???__
'redesignate', 'disaggregate', 'adulterate', 'effectuate', 'prorate', 'subleasing', 'urbanize', 'repurpose'

__???__
'necessary', 'need', 'good', 'able', 'time', 'way', 'know', 'particular', 'kind', 'help', 'example', 'fact'

If the analysis creates a cluster related to the user's interests, these results
could suggest additional terms to extract from the budget. But, I did not
get a complete list of topics (for example, there's no clear border patrol cluster). 
The word butterfly did not appear in what seemed to be the environment-related cluster, nor did any of the 
other animals mentioned in the budget. 

Regularization increased 
the number of words that would cluster, but it also increased the number of clusters that 
were seemingly meaningless and broke down some of the useful clusters into less helpful
subsets. I was unable to tune the regularization without manually reading through the clusters
looking for what did or did not make sense, since there is no reliable quality metric for clustering
English words for human meaning, which is related, but not equivalent to, the 
mathematical closeness in the word vector model. 


### Using word vectors at the document level

Source code: ``word2vec.ipynb``

Gensim provides a distance metric for comparing documents that takes
word similarity into account. It is called the 'Word Mover Distance.'
Each word in each document is matched with one more more words in the 
other document and the distance is weighted based on word similarity 
scores. 

This metric is very slow to compute, so I only used the documents of
length 10 to 100 words (after removing stop words). This represented 942 of
my original 1248 documents.

It's hard to compare results directly with the earlier attempt at clustering
because the data set was shortened.  112 documents grouped into clusters of similarly
structured budget entries that could be eliminated in our search for unusual entries, but
this still left 830 unclustered documents that would have to be analyszed further.
With regularization, the butterfly provision was in a subset of 700 documents. But,
again, without looking at the results manually, it was hard to tell what 
changing alpha was actually accomplishing. 


# Results

While writing and debugging my code, I definitely found the sorts of things that I was looking for (buried amidst plenty of tedious legalese). The authorship committees were clearly concerned with various border and immigration issues, but there were also 
unique provisions benefitting, for example, the sugar beet industry in Oregon. 

There is a potentially interesting political science study 
examining 
what lawmakers consider to be important issues by what 
'strings' they attach to the budget for an individual year or over time. But, so far it would take humans to tag and 
cateogorize the items, as I did not find a 
reliable way to organize the entries in the federal budget using unsupervised machine learning. 

# Future work 

- Creating a word vector model trained on appropriations bills or other legislation to see if it could improve on the clustering by context words tests.  
- Looking at bigrams or trigrams to pick up on euphemisms 'pedestrian fencing' for 'border wall.' I ended up using
department as a budget stopword because there are so many departments in the government, but n-grams would help
distiguish the State Department from the Defense Department for example, from other uses of the word 'state' or 'defense.'
- Using supervised search to filter down the budget based on standard phrases. For example, 'none of the fund apprpriated under this act may be...' is a common phrase introducing a spending restriction. Filtering 
on known phrases or particular headings of interest first and doing some sort of automated analysis from there
may be useful. 
- This project was defined more but what it was not looking for (typical
budget items) than what it was looking for. Creating 
a clear definition of what constitutes
a butterfly-like 'outlier' provision and what doesn't and building a
training set for that would be an interesting task as well. 


