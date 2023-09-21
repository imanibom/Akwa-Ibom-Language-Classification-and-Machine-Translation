import streamlit as st
import pickle
import numpy as np
import pandas as pd
from ibom import make_prediction
from english import trans_prediction
from classify import make_class
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# text preprocessing modules
# text preprocessing modules
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import warnings
# visualiztion tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter



warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

# load stop words
stop_words = stopwords.words("english")

with open("class.pkl", "rb") as file:
    class_model = pickle.load(file)

with open("ibom.pkl", "rb") as file:
    ibibio_model = pickle.load(file)

with open("english.pkl", "rb") as file:
    english_model = pickle.load(file)


st.title('AKWA IBOM LANGUAGES TO ENGLISH LANGUAGE TRANSLATOR')
@st.cache_data
def load_data():
    return pd.read_csv('./ibibio.csv', encoding="unicode_escape")

# Data loading and Processing
df = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

st.sidebar.header("You are Welcome. Enter Akwa Ibom Word or Sentence of Your Choice")

st.title("Translation")
st.write("This is an Akwa Ibom Languages to English Language Translator")

tf = TfidfVectorizer(min_df=0.0,stop_words='english')
le = LabelEncoder()

st.subheader("User Input")

def main():
    # Declare a form to receive a movie's review
    form = st.form(key="my_form")
    review = form.text_input(label="Enter your Akwa Ibom Dialect Word(s)")

    if submit:= form.form_submit_button(label="Translation"):
        _extracted_from_main_8(review)
    form5 = st.form(key="english")
    review5 = form5.text_input(label="Enter English Word(s)")
    if submit:= form5.form_submit_button(label="Ibibio Translation"):
        # make prediction from the input text
        result5= trans_prediction(review5)
        st.title("Ibibio Translation")
        _extracted_from_main_15(result5, review5)


# TODO Rename this here and in `main`
def _extracted_from_main_8(review):
    # make prediction from the input text
    result = make_prediction(review)
    # classify the word
    trans = make_class(review)
    # Display results of the NLP task
    st.title("Translation")
    st.write(result)
    st.title("Dialect")
    _extracted_from_main_15(trans, review)


# TODO Rename this here and in `main`
def _extracted_from_main_15(arg0, arg1):
    st.write(arg0)
    st.title("Use Cases")
    mask = df.dialect.str.contains(arg1)
    st.write(df[["dialect","translation","(language/dialect)"]][mask])

if __name__ == '__main__':
    main()

# Exploring our class description - the categorical variable lang_id
st.title("Explore Data")
st.subheader("Word Cloud")
st.set_option('deprecation.showPyplotGlobalUse', False)
ls = df.dialect.tolist()
mystr = " "
for x in ls:
    mystr += " " + x
if st.button("Word Cloud"):
    w=WordCloud().generate(mystr)
    plt.imshow(w)
    plt.axis('off')
    st.pyplot()

st.subheader("Word Frequency")
if st.button("Word Frequency"):
    counts = dict(Counter(ls).most_common(20))
    labels, values = zip(*counts.items())

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    indexes = np.arange(len(labels))

    bar_width = 0.1

    plt.barh(values, indexes)

    # add labels
    plt.yticks(indexes + bar_width, labels)
    st.pyplot()
    
st.title("ADD NEW WORDS OR SENTENCE")
if st.button("Add Word or Sentence to Database"):
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