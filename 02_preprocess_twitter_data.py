import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary

TWEETS_FN = "data/raw/twitter/2022_01_01_to_07.csv"
OUTPUT_FN = "data/prepared/twitter/2022_01_01_to_07_{}_03.csv"

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
    tokens = tkz.tokenize(text)
    tokens = [t.lower() for t in tokens if t not in stopwords_eng]
    lemmas = [ltz.lemmatize(t) for t in tokens]
    return lemmas


df = pd.read_csv(TWEETS_FN)

print("tweets loaded")
clean_urls(df)
print("urls cleaned, preprocessing")
df['text'] = df['text'].apply(preprocess)
print("saving lemmas to {}".format(OUTPUT_FN.format("lemmas")))
df.to_csv(OUTPUT_FN.format("lemmas"))

print("generating BoW feature vectors. \nbuilding dict.")
text_dict = Dictionary(df.text)
print("saving dict/word indexes to {}".format(OUTPUT_FN.format("dict")))
with open(OUTPUT_FN.format("dict"), 'w') as f:
    for k in text_dict.token2id:
        v = text_dict.token2id[k]
        f.write("{}, {}\n".format(k, v))

print("building bow features")
df['bow_features'] = df['text'].apply(lambda t: text_dict.doc2bow(t))
print("saving bow features to {}".format(OUTPUT_FN.format("bow")))
df = df.drop(columns="text")
df.to_csv(OUTPUT_FN.format("bow"))
