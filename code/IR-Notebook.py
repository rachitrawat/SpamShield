
# coding: utf-8

# ** Objective: ** 
# ** To develop a statistical model for predicting whether questions will be upvoted, downvoted, or closed based on their text. ** 
# ** To predict how long questions will take to answer. **
# 
# ** Authors: Rachit Rawat, Rudradeep Guha, Vineet Nandkishore **

# ** Setup Environment **

# In[1]:


# load required packages

# for creating dataframes from csv datasets
import pandas as pd

# for stripping stop words
from nltk.corpus import stopwords

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

# In[2]:


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


# ** pandas - load CSV into dataframe **
# 

# In[3]:


# Read CSV

# Original Dimensionality - (rows, columns)

# (1264216, 7) 
# Columns (Id, OwnerUserId, CreationDate, ClosedDate, Score, Title, Body)
# frame every 100th question (resource restraints)
questions_df = pd.read_csv(dataset_dir+dataset_dir_questions, encoding='latin1').iloc[::100, :]

# (2014516, 6)
# Columns (Id, OwnerUserId, CreationDate, ParentId, Score, Body)
# frame every 100th answer (resource restraints)
answers_df = pd.read_csv(dataset_dir+dataset_dir_answers, encoding='latin1').iloc[::100, :]

# (3750994, 2)
# Columns (Id, Tag)
# frame every 100th tag (resource restraints)
tags_df = pd.read_csv(dataset_dir+dataset_dir_tags, encoding='latin1').iloc[::100, :]


# ** Sample dataframe before stripping **

# In[4]:


# Calculate dimensionality
# questions_df.shape 
# answers_df.shape 
# tags_df.shape 

# Sample dataframe - uncomment to view
questions_df.head(10) 
# answers_df.head(10)
# tags_df.head(10) 


# ** Strip HTML tags and stop words from text body **

# In[5]:


# Remove HTML tags and stop words from body and title column
for index, row in questions_df.iterrows():
   questions_df.at[index, 'Body']= ' '.join([word for word in strip_tags(row[6]).split() if word not in cachedStopWords])
   questions_df.at[index, 'Title']= ' '.join([word for word in strip_tags(row[5]).split() if word not in cachedStopWords])

# Remove HTML tags and stop words from body column
for index, row in answers_df.iterrows():
   answers_df.at[index, 'Body']= ' '.join([word for word in strip_tags(row[5]).split() if word not in cachedStopWords]) 


# ** Sample dataframe after stripping **

# In[6]:


# Calculate dimensionality
# questions_df.shape 
# answers_df.shape 
# tags_df.shape 

# Sample dataframe - uncomment to view
questions_df.head(10)
# answers_df.head(10)
# tags_df.head(10)


# # Initial analysis

# ** Top 10 most common tags **

# In[7]:


tags_tally = collections.Counter(tags_df['Tag'])

# x = tag name, y = tag frequency
x, y = zip(*tags_tally.most_common(10))

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colors = [colormap(i) for i in np.linspace(0, 0.8,50)]   

area = [i/50 for i in list(y)]   # 0 to 15 point radiuses
plt.figure(figsize=(8,8))
plt.ylabel("Frequency")
plt.title("Top 10 most common tags")
for i in range(len(y)):
        plt.plot(i,y[i], marker='v', linestyle='',ms=area[i],label=x[i])

plt.legend(numpoints=1)
plt.show()


# ![](http://)![](http://)**Distribution  - number of answers per question**

# In[8]:


ans_per_question = collections.Counter(answers_df['ParentId'])
answerid,noAnswers= zip(*ans_per_question.most_common())

N=50
plt.bar(range(N), noAnswers[:N], align='center', alpha=0.7)
#plt.xticks(y_pos, objects)

plt.ylabel('Number of Answers per Question')
plt.xlabel('Question Id')
plt.title('Distribution of Answers per question ')
plt.text(10,4,"Average answers per question: "+str(math.floor((np.mean(noAnswers)))))

plt.show()

