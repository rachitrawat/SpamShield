import pandas as pd

# read csv
sms = pd.read_csv('C:\\Users\\rachit\\Documents\\spam.csv', encoding='latin-1')
# drop unnamed columns
sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
# rename columns
sms = sms.rename(columns = {'v1':'class','v2':'message'})
# add msg len feature
sms['length'] = sms['message'].apply(len)
