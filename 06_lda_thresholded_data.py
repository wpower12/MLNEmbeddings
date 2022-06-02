import pandas as pd
from gensim.models.ldamodel import LdaModel
from mlnhelper import util

NUM_TOPICS = 50
ID = "v00"

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

# <ID, Threshold, 'type' (bow, dict, topics)>
DATA_FN_STUB = "2022_01_01_to_07_{}_{}_{}.csv"
DATA_DIR = "data/prepared/twitter/thresholded"

for threshold in THRESHOLDS:
    FN_BOW  = "{}/{}".format(DATA_DIR, DATA_FN_STUB.format(ID, threshold, "bow"))
    FN_DICT = "{}/{}".format(DATA_DIR, DATA_FN_STUB.format(ID, threshold, "dict"))
    FN_OUT  = "{}/{}".format(DATA_DIR, DATA_FN_STUB.format(ID, threshold, "topics"))

    print("loading data: {}".format(threshold))
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
                          num_topics=NUM_TOPICS, # This doesn't seem to be working?
                          id2word=text_dict,
                          random_state=2,
                          alpha="auto",
                          passes=1)

    print("saving topics to {}".format(FN_OUT))
    with open("{}".format(FN_OUT), 'w') as f:
        # I think we see all topics now. 
        for topic in tweets_lda.show_topics(num_topics=NUM_TOPICS, formatted=True):
            f.write("{}\n".format(topic))
