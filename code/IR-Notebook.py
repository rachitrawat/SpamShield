
# coding: utf-8

# ## Setup Environment

# In[ ]:

# load required packages

# for creating dataframes from csv datasets
import pandas as pd

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

