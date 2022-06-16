"""

"""
import pandas as pd
from mlnhelper import utils

DATA_DIR = "data/prepared/twitter"

FN_USERID_2_FIPS   = "2022_01_01_to_07_userid_map_02.csv"
FN_BOWID_2_TERM    = "2022_01_01_to_07_v00_keep_dict.csv"
FN_TWEET_BOW       = "2022_01_01_to_07_v00_keep_bow.csv"
FN_USER_AGGED_BOWS = "2022_01_01_to_07_v00_agged_bow.csv"  # Userid -> dict of {bow_term_id -> weight}

# Nodes - create the node features, with type-one-hot feature vectors. Using 3 maps to keep track of the node_ids
#         for each type of node: fips_2_idx, user_2_idx, bowid_2_idx. All of which resolve a type-identifier to
#         a node index.
nodes = []
counties = set()
user_2_fips = {}
with open("{}/{}".format(DATA_DIR, FN_USERID_2_FIPS)) as f_userid_2_fips:
    for line in f_userid_2_fips:
        userid, fips = line.replace("\n", "").split(",")
        user_2_fips[userid] = fips
        counties.add(fips)

bowid_2_term = {}
with open("{}/{}".format(DATA_DIR, FN_BOWID_2_TERM)) as f_bow_2_term:
    for line in f_bow_2_term:
        term, bowid = line.replace("\n", "").split(", ")
        bowid_2_term[bowid] = term

# We use the two maps derived above to generate the consistent node indexes, while we also create the feature vectors.
fips_2_idx = {}
fips_n_idx = 0
for county in counties:
    fips_2_idx[county] = fips_n_idx
    nodes.append([1, 0, 0])
    fips_n_idx += 1

user_2_idx = {}
user_n_idx = fips_n_idx+1
for user in user_2_fips:
    user_2_idx[user] = user_n_idx
    nodes.append([0, 1, 0])
    user_n_idx += 1

bowid_2_idx = {}
bowid_n_idx = user_n_idx+1
for bowid in bowid_2_term:
    bowid_2_idx[bowid] = bowid_n_idx
    nodes.append([0, 0, 1])
    bowid_n_idx += 1

# Edges -
#   User-County - Using the user_2_fips dict.
#   User-Term   - Using an aggregate of the bows?
#   County-County - Need the fips adjacecy list.
edges = []

for user in user_2_fips:
    fips = user_2_fips[user]
    user_n_idx = user_2_idx[user]
    fips_n_idx = fips_2_idx[fips]
    edges.append((user_n_idx, fips_n_idx))
    edges.append((fips_n_idx, user_n_idx))

# Need to 'load' the agged_bows
user_2_bows = {}
with open("{}/{}".format(DATA_DIR, FN_USER_AGGED_BOWS)) as f_bows:
    f_bows.readline()
    for line in f_bows.readlines():
        userid, bow_str = line.replace("\n", "").split(",", 1)
        user_2_bows[userid] = util.str_to_dict(bow_str)

for user in user_2_bows:
    bow_dict = user_2_bows[user]
    user_n_idx = user_2_idx[user]

    for bow_id in bow_dict:
        weight = bow_dict[bow_id]
        bowid_n_idx = bowid_2_idx[bow_id]

        edges.append((user_n_idx, bowid_n_idx))
        edges.append((bowid_n_idx, user_n_idx))

print(len(nodes), len(edges))
