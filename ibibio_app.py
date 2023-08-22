import streamlit as st
import pickle
import numpy as np
import pandas as pd
from ibom import make_prediction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import joblib
import warnings

warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

# load stop words
stop_words = stopwords.words("english")



with open("ibom.pkl", "rb") as file:
    model = pickle.load(file)

st.title('ENGLISH-AKWA IBOM LANGUAGES TRANSLATOR')
@st.cache_data
def load_data():
    return pd.read_csv('./ibibio.csv', encoding="unicode_escape")

# Data loading and Processing
df = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

st.sidebar.header("You are Welcome. Enter English Word or Sentence of Your Choice")

st.title("Translation")
st.write("This is an English to Akwa Ibom Languages Translator")

tf = TfidfVectorizer(min_df=0.0,stop_words='english')
le = LabelEncoder()

st.subheader("User Input")

# Declare a form to receive a movie's review
form = st.form(key="my_form")
review = form.text_input(label="Enter your English Word(s)")

if submit:= form.form_submit_button(label="Translation"):
    # make prediction from the input text
    result = make_prediction(review)
    # Display results of the NLP task
    st.title("Translation")
    st.write(result)
    st.title("Use Cases")
    mask = df.translation.str.contains(review)
    st.write(df[["translation","dialect","(language/dialect)"]][mask])
    
    
st.title("Add Word or Sentence to Database")
form2 = st.form(key="new_words")
review2 = form2.text_input(label="Enter English Word or Sentence")

if submit2:= form2.form_submit_button(label="Add Translation"):
    st.write(review2)
form3 = st.form(key="translate")
review3 = form3.text_input(label="Enter Translation")
if submit3:= form3.form_submit_button(label="Enter language/Dialect"):
    st.write(review3)
form4 = st.form(key="language_dialect")
review4 = form4.text_input(label="Enter Language/Dialect")
if submit4:= form4.form_submit_button(label="Added language/Dialect"):
    st.write(review4)
# Concat dataframes
# Define the new row to be added
t2 = review2
t3 = review3
t4 = review4
new_row = pd.DataFrame([{"dialect": t3,"translation": t2, "(language/dialect)": t4}])    
# Use the loc method to add the new row to the DataFrame
df = pd.concat([df, new_row], ignore_index= True)
# saving the dataframe
df.to_csv('./ibom.csv')