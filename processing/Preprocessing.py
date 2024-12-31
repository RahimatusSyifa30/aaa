import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd

# Case folding
def casefolding(content):
    return content.lower()

# Cleansing
def cleansing(content):
    content = [str(entry) for entry in content]
    content = [entry.strip() for entry in content]
    content = [re.sub(r'[?|$|.|!_:/\@%°")(-+,]', '', entry) for entry in content]
    content = [re.sub(r'\d+', '', entry) for entry in content]
    content = [re.sub(r"\b[a-zA-Z]\b", "", entry) for entry in content]
    content = [re.sub(r'[\U0001F600-\U0001F64F]', '', entry) for entry in content]
    content = [re.sub(r'[\u00e0-\u00fc]', '', entry) for entry in content]
    content = [re.sub('\s+', ' ', entry) for entry in content]
    return content

# Tokenisasi
def word_tokenize_wrapper(text):
    return word_tokenize(text)

# Stopword removal
def stopword_removal(content):
    filtering = stopwords.words('indonesian')
    sw = pd.read_csv("D:/Documents/MATKUL SEM 7/PROPOSAL/PELABELAN/stopwords-id.csv")
    filtering.extend(sw)
    def myFunc(x):
        return x not in filtering
    fit = filter(myFunc, content)
    return list(fit)

# Stemming
def stemming(content):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = [stemmer.stem(w) for w in content]
    return " ".join(do)

# Preprocessing pipeline
def preprocess_data(content_series):
    content_series = content_series.apply(casefolding)
    content_series = cleansing(content_series)
    content_series = content_series.apply(word_tokenize_wrapper)
    content_series = content_series.apply(stopword_removal)
    content_series = content_series.apply(stemming)
    return content_series
