import pandas as pd
from gensim.models.ldamodel import LdaModel
from mlnhelper import utils

DATA_FN_STUB = "2022_01_01_to_07_{}_02.csv"
OUTPUT_FN = "2022_01_01_to_07_TOPICS_03.csv"
NUM_TOPICS = 50

DATA_DIR = "data/prepared/twitter"
FN_BOW  = "{}/{}".format(DATA_DIR, DATA_FN_STUB.format("bow"))
FN_DICT = "{}/{}".format(DATA_DIR, DATA_FN_STUB.format("dict"))

print("loading data")
df_bow = pd.read_csv(FN_BOW)
df_bow.drop(df_bow[df_bow['bow_features'] == "[]"].index, inplace=True) # Drop 'empty' tweets
df_bow['bow_features'] = df_bow['bow_features'].apply(util.str_to_tuplelist)

text_dict = {}
with open(FN_DICT, 'r') as f:
    for line in f.readlines():
        raw = line.replace("\n", "", ).split(", ")
        if len(raw) == 2:
            k, v = raw
            text_dict[int(v)] = k  # Because we need id -> string

print("performing LDA")
tweets_lda = LdaModel(df_bow['bow_features'].to_list(),
                      num_topics=NUM_TOPICS,
                      id2word=text_dict,
                      random_state=1,
                      alpha="auto",
                      passes=10)

print("saving topics to {}".format(OUTPUT_FN.format("topics")))
with open("{}/{}".format(DATA_DIR, OUTPUT_FN), 'w') as f:
    topics = tweets_lda.show_topics()
    for topic in topics:
        f.write("{}\n".format(topic))
