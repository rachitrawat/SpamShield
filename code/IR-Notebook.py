
# coding: utf-8

# ## Setup Environment

# In[1]:

# load required packages

# for creating dataframes from csv datasets
import pandas as pd

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


# ## pandas - load CSV into dataframe

# In[ ]:

questions_df = pd.read_csv(dataset_dir+dataset_dir_questions, encoding='latin1')
answers_df = pd.read_csv(dataset_dir+dataset_dir_answers, encoding='latin1')
tags_df = pd.read_csv(dataset_dir+dataset_dir_tags, encoding='latin1')

# Sample dataframe - uncomment to view
# questions_df
# answers_df
# tags_df


# ## Top 10 most common tags

# In[ ]:

tags_tally = collections.Counter(tags_df['Tag'])

# x = tag name, y = tag frequency
x, y = zip(*tags_tally.most_common(10))

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colors = [colormap(i) for i in np.linspace(0, 0.8,50)]   

area = [i/5000 for i in list(y)]   # 0 to 15 point radiuses
plt.figure(figsize=(8,8))
plt.ylabel("Frequency")
plt.title("Top 10 most common tags")
for i in range(len(y)):
        plt.plot(i,y[i], marker='v', linestyle='',ms=area[i],label=x[i])

plt.legend(numpoints=1)
plt.show()

