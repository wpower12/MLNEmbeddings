import pandas as pd
from mlnhelper import utils, pipeline

IN_DIR  = "data/prepared/twitter/thresholded"
FN_STUB = "2022_01_01_to_07_v00_{}_{}"
FN_CADJ = "data/raw/county_adj.txt"
FN_USERID_2_FIPS = "data/prepared/twitter/2022_01_01_to_07_userid_map_02.csv"

THRESHOLDS = [0.01]

stopwords_eng = utils.load_augmented_stopwords()
userid_2_fips, counties = utils.read_user_map(FN_USERID_2_FIPS)
cadj_map = utils.cadj_from_txt(FN_CADJ)

for thresh in THRESHOLDS:
    # load data frames
    print("processing threshold: {}".format(thresh))
    fn_lemmas = "{}/{}.csv".format(IN_DIR, FN_STUB.format(thresh, "lemmas"))
    fn_topics = "{}/{}.csv".format(IN_DIR, FN_STUB.format(thresh, "topics"))
    fn_viz    = "{}/{}.png".format(IN_DIR, FN_STUB.format(thresh, "embedding"))
    pipeline.twitter_mln_embedding_viz_pipeline(fn_lemmas,
                                                fn_topics,
                                                userid_2_fips,
                                                counties,
                                                fn_viz,
                                                stopwords=stopwords_eng,
                                                cadj_map=cadj_map)
