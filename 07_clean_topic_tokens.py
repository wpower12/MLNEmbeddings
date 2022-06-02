"""
Reads in a _topics file. Gathers a set of all the tokens found for each topic from the _topics file. Then reads in a
_lemma file. Saves _keep_lemmas, _keep_bow file, _keep_dict file that only contains tweets with those tokens, and a bow
vector (and index) that only considers the found topic-terms/tokens.
"""
import pandas as pd
import string
from nltk.corpus import stopwords
from gensim.corpora import Dictionary

stopwords_eng = stopwords.words("english")
stopwords_eng += string.punctuation
stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
stopwords_eng += [', ', "’, ’", '’, ’', '’, ’', '\'', "'", "`", "’"]

DATA_DIR = "data/prepared/twitter"
FN_TOPICS = "2022_01_01_to_07_TOPICS_03.csv"
FN_LEMMAS = "2022_01_01_to_07_lemmas_02.csv"
FN_TOPICS = "{}/{}".format(DATA_DIR, FN_TOPICS)
FN_LEMMAS = "{}/{}".format(DATA_DIR, FN_LEMMAS)

FN_OUT_STUB = "2022_01_01_to_07_v00_{}.csv"
FN_K_BOW  = "{}/{}".format(DATA_DIR, FN_OUT_STUB.format("keep_bow"))
FN_K_DICT = "{}/{}".format(DATA_DIR, FN_OUT_STUB.format("keep_dict"))

with open(FN_TOPICS, "r") as f_topics:
    terms = set()
    for topic in f_topics.readlines():
        term_str = topic.split(", ")[1]
        topic_terms = term_str.split(" + ")
        topic_terms = [s.replace("\n", "") for s in topic_terms]
        # topic_terms = [s[7:-1] for s in topic_terms]
        topic_terms = [s.split("*")[1][1:-1].replace("'", "") for s in topic_terms]
        for t in topic_terms:
            if t not in stopwords_eng:
                terms.add(t)

terms.add('this')
# print(terms)
count = 0


def clean_lemmas(lemma_str):
    global count
    raw_lemmas = lemma_str.replace("]", "").replace("[", "").split(", ")
    good_lemmas = []
    for l in raw_lemmas:
        l = l.replace("'", "")
        if l in terms and len(l) > 0:
            good_lemmas.append(l)
    if len(good_lemmas) > 0:
        count += 1
    return good_lemmas


df_lemmas = pd.read_csv(FN_LEMMAS, index_col=0, header=0)

df_c = df_lemmas.head().copy()
df_lemmas['lemmas'] = df_lemmas['text'].apply(clean_lemmas)
# df_c['lemmas'] = df_c['text'].apply(clean_lemmas)
print(count)

text_dict = Dictionary(df_lemmas['lemmas'])
df_lemmas['bow'] = df_lemmas['lemmas'].apply(lambda l: text_dict.doc2bow(l))
print(df_lemmas.head())
print(df_lemmas['text'].head())

df_lemmas = df_lemmas.drop(columns=["text", "lemmas"])
df_lemmas.to_csv(FN_K_BOW)
with open(FN_K_DICT, 'w') as f:
    for k in text_dict.token2id:
        v = text_dict.token2id[k]
        f.write("{}, {}\n".format(k, v))
