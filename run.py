# import libraries
import string
from tkinter import *
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# read csv

sms = pd.read_csv('C:\\Users\\rachit\\PycharmProjects\\CS309-IR-Monsoon-2017-RR\\spam.csv', encoding='latin-1')
sms = sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# make a copy of message column
text_feat = sms['message'].copy()


# remove stop words, punctuation and do stemming
def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i)) + " "
    return words


# preprocessing
text_feat = text_feat.apply(text_process)


def predict(msg_str):
    # vectorization
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)

    # split features to test and training set
    features_train, features_test, labels_train, labels_test = train_test_split(features, sms['class'], test_size=0.1,
                                                                                random_state=111)

    from sklearn.naive_bayes import MultinomialNB

    # initialize MNB
    mnb = MultinomialNB(alpha=0.2)

    # functions to fit our classifiers and make predictions
    def train_classifier(clf, feature_train, labels_train):
        clf.fit(feature_train, labels_train)

    train_classifier(mnb, features_train, labels_train)

    def test_case(m):  # m is the text string
        test_data = pd.Series(m, name='message')
        test_data = test_data.apply(text_process)
        test_feature = vectorizer.transform(test_data)
        return test_feature

    pred_val = (mnb.predict(test_case(msg_str)))[0]
    if pred_val == "ham":
        pred_val = "Not a spam"
    return pred_val


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Spam Shield")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)


root = Tk()
e = Entry(root)

text = Label(text="Enter message to check for spam: ")
text.config(font=("Courier", 15))
text.pack()

text1 = Label(text="")
text1.config(font=("Courier", 15))

# size of the window
root.geometry(str(root.winfo_screenwidth()) + "x" + str(root.winfo_screenheight()))

e.pack()
e.focus_set()


def checkForSpam():
    text1.config(text=predict(e.get()))
    text1.pack()

b = Button(root, text='Check', command=checkForSpam)
b.pack(side='top')
app = Window(root)
root.mainloop()
