import pandas as pd
import numpy as np
import string
import re
import nltk
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import SGDClassifier
from collections import Counter
import nlpaug.augmenter.word as naw
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
import pickle

# Data loading and Processing
df = pd.read_csv('./ibibio.csv', encoding="unicode_escape")

stop_words=stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_data(text):
    text=text.lower() #lower the text
    text = re.sub(r'[^\w\s]', '', text) #remove irrelevant characters    
    text = text.split() #convert sentence to tokens
    text = [lemmatizer.lemmatize(word) for word in text] #lemmatization
    text = " ".join(text) #converting tokens to sentence
    return text

df["dialect"] = df["dialect"].apply(clean_data)

# Modelling
X = df['translation']
y = df['dialect']

le = LabelEncoder()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=100,test_size=0.2)

tf = TfidfVectorizer(ngram_range=(1, 3),min_df=0.0,stop_words='english')

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)

model = SGDClassifier(n_jobs=-1,random_state=100,loss='modified_huber',alpha=0.0005)
model.fit(X_train_tf,y_train)

y_pred = model.predict(X_test_tf)

labels = np.unique(y_test)
ytest_prob = label_binarize(y_test, classes=labels)
ypred_prob = label_binarize(y_pred, classes=labels)


def make_prediction(questn):
    clean_ques = clean_data(questn)
    clean_ques = tf.transform([clean_ques])
    if np.amax(model.predict_proba(clean_ques)):
        return le.inverse_transform(model.predict(clean_ques))[:]
    else:
        return (questn," is not yet in our register")
    
#w = input("enter word(s): ")
#print(make_prediction(w))
#print(df.head())

with open("ibom.pkl", "wb") as file:
    pickle.dump(model, file)