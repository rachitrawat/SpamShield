import tkinter as tk
import string
from PIL import Image, ImageTk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pygame import mixer
# read csv

sms = pd.read_csv('spam.csv', encoding='latin-1')
sms = sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# make a copy of message column
text_feat = sms['message'].copy()
mixer.init()

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
        mixer.music.load('ham.mp3')
        mixer.music.play()
        pred_val = "Not a spam"
    else:
        mixer.music.load('spam.mp3')
        mixer.music.play()
    return pred_val


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

    def show(self):
        self.lift()


class Page1(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        Page1.spam_list = []
        Page1.not_spam_list = []
        #        self.configure(background='black')
        image = Image.open("1.jpg")
        bg_image = ImageTk.PhotoImage(image)
        bg_label = tk.Label(self, image=bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_image

        e = tk.Entry(self, width=50, justify='center')

        text = tk.Label(self, text="Enter message to check for spam: ")
        text.config(font=("Times New Roman", 25), fg='white', bg='black')
        text.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        text1 = tk.Label(self, text="")
        text1.config(font=("Museo Sans", 15, "bold"), fg='white', bg='black')
        text1.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

        e.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
        e.config(fg='black', bg='grey')
        e.focus_set()

        def checkForSpam():
            text1.config(text=predict(e.get()))
            if "N" in text1.cget("text"):
                self.not_spam_list.append(e.get())
                Page2.label.config(text=("\n".join(self.not_spam_list)), fg='white', bg='black',
                                   font=("Museo Sans", 15, "bold"))
            else:
                self.spam_list.append(e.get())
                Page3.label.config(text=("\n".join(self.spam_list)), fg='white', bg='black',
                                   font=("Museo Sans", 15, "bold"))

        b = tk.Button(self, text='Check', command=checkForSpam)
        b.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        b.config(width=15, height=2, fg='white', bg='black')


class Page2(Page1):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        #        image = Image.open("2.jpg")
        #        bg_image = ImageTk.PhotoImage(image)
        #        bg_label = tk.Label(self, image=bg_image)
        #        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        #        bg_label.image = bg_image
        Page2.label = tk.Label(self, text="")
        Page2.label.pack(side="top", fill="both", expand=True)


class Page3(Page1):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        #        image = Image.open("1.jpg")
        #        bg_image = ImageTk.PhotoImage(image)
        #        bg_label = tk.Label(self, image=bg_image)
        #        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        #        bg_label.image = bg_image
        Page3.label = tk.Label(self, text="")
        Page3.label.pack(side="top", fill="both", expand=True)


class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(root, text="Home", command=p1.lift)
        b1.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        b1.config(width=10, height=2, fg='gold', bg='black')
        b2 = tk.Button(root, text="History of Not Spam", command=p2.lift)
        b2.place(relx=0.2, rely=0.15, anchor=tk.CENTER)
        b2.config(width=20, height=2, fg='light green', bg='black')
        b3 = tk.Button(root, text="History of Spam", command=p3.lift)
        b3.place(relx=0.8, rely=0.15, anchor=tk.CENTER)
        b3.config(width=20, height=2, fg='red', bg='black')

        p1.show()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("SpamShield")
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("500x500")
    root.mainloop()
