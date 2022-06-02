import pandas as pd
import numpy as np
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary

ID = "v00"

DATA_FN_STUB = "2022_01_01_to_07_thresh_{}.csv"
DATA_DIR = "data/raw/twitter/thresholded"

# parameters: ID, Threshold, 'kind'
OUT_FN_STUB = "2022_01_01_to_07_{}_{}_{}.csv"
OUT_DIR = "data/prepared/twitter/thresholded"

# THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
THRESHOLDS = [0.5, 1.0]

stopwords_eng = stopwords.words("english")
stopwords_eng += string.punctuation
stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()


# 1 - Remove URLs
def clean_urls(dframe):
    dframe['text'] = dframe['text'].str.replace(r"http\S+", "")


# 2 - Cleans Stop Words
# 3 - Tokenize
# 4 - Lemmatize
def preprocess(text):
    if(text != None):
        tokens = tkz.tokenize(text)
        tokens = [t.lower() for t in tokens if t not in stopwords_eng]
        lemmas = [ltz.lemmatize(t) for t in tokens]
        return lemmas
    return []


for threshold in THRESHOLDS:
    print("threshold {} {}".format(threshold,  DATA_FN_STUB.format(threshold)))
    df = pd.read_csv("{}/{}".format(DATA_DIR, DATA_FN_STUB.format(threshold)),
        engine="python",
        header=0,
        index_col=0,
        dtype={"userid": str, "text": str})

    print("tweets loaded")
    print(df.head())
    clean_urls(df)
    print("urls cleaned, preprocessing")
    df['text'] = df['text'].apply(preprocess)
    fn_lemma = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format(ID, threshold, "lemmas"))
    print("saving lemmas to {}".format(fn_lemma))
    df.to_csv(fn_lemma)
    
    print("generating BoW feature vectors. \nbuilding dict.")
    text_dict = Dictionary(df.text)
    fn_dict = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format(ID, threshold, "dict"))
    print("saving dict/word indexes to {}".format(fn_dict))
    with open(fn_dict, 'w') as f:
        for k in text_dict.token2id:
            v = text_dict.token2id[k]
            f.write("{}, {}\n".format(k, v))
    
    print("building bow features")
    df['bow_features'] = df['text'].apply(lambda t: text_dict.doc2bow(t))
    fn_bow = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format(ID, threshold, "bow"))
    print("saving bow features to {}".format(fn_bow))
    df = df.drop(columns="text")
    df.to_csv(fn_bow)
