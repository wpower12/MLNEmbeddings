from mlnhelper import networks, embedding

DATA_DIR = "data/prepared/twitter"
FN_USERID_2_FIPS   = "{}/{}".format(DATA_DIR, "2022_01_01_to_07_userid_map_02.csv")
FN_BOWID_2_TERM    = "{}/{}".format(DATA_DIR, "2022_01_01_to_07_v00_keep_dict.csv")
FN_TWEET_BOW       = "{}/{}".format(DATA_DIR, "2022_01_01_to_07_v00_keep_bow.csv")
FN_USER_AGGED_BOWS = "{}/{}".format(DATA_DIR, "2022_01_01_to_07_v00_agged_bow.csv")
FN_CADJ = "data/raw/county_adj.txt"


nodes, edges, maps = networks.build_twitter_mln_from_raws(FN_USERID_2_FIPS,
                                                          FN_BOWID_2_TERM,
                                                          FN_USER_AGGED_BOWS,
                                                          fn_county_adj=FN_CADJ)

embeddings = embedding.learn_embeddings(nodes, edges)
low_dim_rep_df = embedding.tsne(embeddings)

