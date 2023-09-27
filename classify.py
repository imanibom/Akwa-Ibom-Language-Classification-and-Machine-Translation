import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pickle

# Data loading and Processing
df = pd.read_csv('./ibibio.csv', encoding="unicode_escape")
df = df.astype(str)
df = df.dropna()

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
X = df['dialect']
y = df['(language/dialect)']

le = LabelEncoder()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=100,test_size=0.3)

tf = TfidfVectorizer(ngram_range=(1, 50),min_df=0.0,stop_words='english')

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)

model = SGDClassifier(n_jobs=-1,random_state=100,loss='modified_huber',alpha=0.00005)

model.fit(X_train_tf,y_train)

# prediction
y_pred = model.predict(X_test_tf)

labels = np.unique(y_test)
ytest_prob = label_binarize(y_test, classes=labels)
ypred_prob = label_binarize(y_pred, classes=labels)


def make_class(questn):
    clean_ques = clean_data(questn)
    clean_ques = tf.transform([clean_ques])
    if np.amax(model.predict_proba(clean_ques)):
        return le.inverse_transform(model.predict(clean_ques))[0]
    else:
        return (questn," is not yet in our register")
    
# save as pickle file
with open("class.pkl", "wb") as file:
    pickle.dump(model, file)