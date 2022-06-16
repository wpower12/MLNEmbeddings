from mlnhelper import utils
import pandas as pd

DATA_DIR = "data/prepared/twitter"
FN_USERID_2_FIPS = "2022_01_01_to_07_userid_map_02.csv"
FN_BOWID_2_TERM  = "2022_01_01_to_07_v00_keep_dict.csv"
FN_TWEET_BOW     = "2022_01_01_to_07_v00_keep_bow.csv"

FN_OUT = "2022_01_01_to_07_v00_agged_bow.csv"

user_2_bows = {}
df_bow = pd.read_csv("{}/{}".format(DATA_DIR, FN_TWEET_BOW))
for index, row in df_bow.iterrows():
    bow_str = row['bow']
    userid  = row['userid']
    bow_tuples = util.str_to_tuplelist(bow_str)

    if userid not in user_2_bows:
        user_2_bows[userid] = {}

    # Just going to sum the bows for now.
    user_bow_dict = user_2_bows[userid]
    for entry in bow_tuples:
        bow_id, _ = entry
        if bow_id not in user_bow_dict:
            user_bow_dict[bow_id] = 1
        else:
            user_bow_dict[bow_id] += 1

with open("{}/{}".format(DATA_DIR, FN_OUT), "w") as f_out:
    for user in user_2_bows:
        f_out.write("{},{}\n".format(user, user_2_bows[user]))
