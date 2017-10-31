
# coding: utf-8

# ** Objective: ** 
# ** To develop a statistical model for predicting whether questions will be upvoted, downvoted, or closed based on their text. ** 
# ** To predict how long questions will take to answer. **
# 
# ** Authors: Rachit Rawat, Rudradeep Guha, Vineet Nandkishore **

# ** Setup Environment **

# In[ ]:


# load required packages

# for creating dataframes from csv datasets
import pandas as pd

# for regular expressions
import re

# for stripping stop words
from nltk.corpus import stopwords

# for TF-IDF
from textblob import TextBlob as tb

# for removing HTML tags from text body
from html.parser import HTMLParser

# for counting
import collections

# for scientific computing
import numpy as np
import math

# for plotting graphs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# magic function
get_ipython().magic('matplotlib inline')

# kaggle - data set files are available in the "../input/" directory
dataset_dir = "../input/"
dataset_dir_questions = "Questions.csv"
dataset_dir_answers = "Answers.csv"
dataset_dir_tags = "Tags.csv"

# for offline run
# dataset_dir = "/home/su/Downloads/stacksample"

# list the files in the dataset directory
from subprocess import check_output
print(check_output(["ls", dataset_dir]).decode("utf8"))

cachedStopWords = stopwords.words("english")



# ** HTML tags Stripper class **

# In[ ]:


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# ** TF-IDF helper fucntions **

# In[ ]:


# tf(word, blob) computes "term frequency" which is the number of times 
# a word appears in a document blob,normalized by dividing by 
# the total number of words in blob. 
# We use TextBlob for breaking up the text into words and getting the word counts.
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

# n_containing(word, bloblist) returns the number of documents containing word.
# A generator expression is passed to the sum() function.
def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

# idf(word, bloblist) computes "inverse document frequency" which measures how common 
# a word is among all documents in bloblist. 
# The more common a word is, the lower its idf. 
# We take the ratio of the total number of documents 
# to the number of documents containing word, then take the log of that. 
# Add 1 to the divisor to prevent division by zero.
def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

# tfidf(word, blob, bloblist) computes the TF-IDF score. 
# It is simply the product of tf and idf.
def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


# # 1. Preprocessing
# **1.1 pandas - load CSV into dataframe **
# 

# In[ ]:


# Read CSV

# Original Dimensionality - (rows, columns)

# (1264216, 7) 
# Columns (Id, OwnerUserId, CreationDate, ClosedDate, Score, Title, Body)
# frame every 1000th question (resource restraints)
questions_df = pd.read_csv(dataset_dir+dataset_dir_questions, encoding='latin1').iloc[::10000, :]

# (2014516, 6)
# Columns (Id, OwnerUserId, CreationDate, ParentId, Score, Body)
# frame every 1000th answer (resource restraints)
answers_df = pd.read_csv(dataset_dir+dataset_dir_answers, encoding='latin1').iloc[::1000, :]

# (3750994, 2)
# Columns (Id, Tag)
# frame every 1000th tag (resource restraints)
tags_df = pd.read_csv(dataset_dir+dataset_dir_tags, encoding='latin1').iloc[::1000, :]


# **1.2 Sample dataframe before stripping **

# In[ ]:


# Calculate dimensionality
# questions_df.shape 
# answers_df.shape 
# tags_df.shape 

# Sample dataframe - uncomment to view
questions_df.head(10) 
# answers_df.head(10)
# tags_df.head(10) 


# **1.3 Strip HTML tags, stop words and symbols from text body and convert to lowercase **

# In[ ]:


# Remove HTML tags, stop words, symbols from body and title column and convert to lowercase
for index, row in questions_df.iterrows():
   questions_df.at[index, 'Body']= ' '.join([word for word in re.sub(r'[^\w]', ' ', strip_tags(row[6])).lower().split() if word not in cachedStopWords])
   questions_df.at[index, 'Title']= ' '.join([word for word in re.sub(r'[^\w]', ' ', strip_tags(row[5])).lower().split() if word not in cachedStopWords])

# Remove HTML tags, stop words, symbols from body and convert to lowercase
for index, row in answers_df.iterrows():
   answers_df.at[index, 'Body']= ' '.join([word for word in re.sub(r'[^\w]', ' ', strip_tags(row[5])).lower().split() if word not in cachedStopWords]) 


# **1.4 Sample dataframe after stripping **

# In[ ]:


# Calculate dimensionality
# questions_df.shape 
# answers_df.shape 
# tags_df.shape 

# Sample dataframe - uncomment to view
questions_df.head(10)
# answers_df.head(10)
# tags_df.head(10)


# ** 1.5 Make a TF-IDF word dictionary **

# In[ ]:


tfidf_dict={}
bloblist=[]
idlist=[]

for index, row in questions_df.iterrows():
    # also append title to text body
    bloblist.append(tb(row[6]+" "+row[5]))
    idlist.append(row[0])

for i, blob in enumerate(bloblist):
    print("Top words in question ID {}".format(idlist[i]))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        tfidf_dict[word]=[round(score, 5), idlist[i]]


# **1.6 Sample [TF-IDF, ID] dictionary **

# In[ ]:


for k, v in tfidf_dict.items():
    print(k, v)


# In[ ]:


def predict(rawQ):
    # strip stop words, symbols and convert to lowercase
    strippedQ= ' '.join([word for word in re.sub(r'[^\w]', ' ', rawQ).lower().split() if word not in cachedStopWords])
    termList=strippedQ.split()
    
    print(termList)
    


# In[ ]:


inputQ="How to delete a table in SQL?"
predict(inputQ)


# # Initial analysis

# ** Top 10 most common tags **

# In[ ]:


tags_tally = collections.Counter(tags_df['Tag'])

# x = tag name, y = tag frequency
x, y = zip(*tags_tally.most_common(10))

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colors = [colormap(i) for i in np.linspace(0, 0.8,50)]   

area = [i/3 for i in list(y)]   # 0 to 15 point radiuses
plt.figure(figsize=(8,8))
plt.ylabel("Frequency")
plt.title("Top 10 most common tags")
for i in range(len(y)):
        plt.plot(i,y[i], marker='v', linestyle='',ms=area[i],label=x[i])

plt.legend(numpoints=1)
plt.show()


# ![](http://)![](http://)**Distribution  - number of answers per question**

# In[ ]:


ans_per_question = collections.Counter(answers_df['ParentId'])
answerid,noAnswers= zip(*ans_per_question.most_common())

N=50
plt.bar(range(N), noAnswers[:N], align='center', alpha=0.7)
#plt.xticks(y_pos, objects)

plt.ylabel('Number of Answers per Question')
plt.xlabel('Question Id')
plt.title('Distribution of Answers per question ')
plt.text(10,1.5,"Average answers per question: "+str(math.floor((np.mean(noAnswers)))))

plt.show()

