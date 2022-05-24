ACTIVE_SUB_FN = "data/raw/reddit/raw_reddit_00/activesubreddits_yd.csv"
COUNTY_FN     = "data/raw/reddit/raw_reddit_00/county.csv"
SUBREDDIT_FN  = "data/raw/reddit/raw_reddit_00/subreddit.csv"
USERYD_FN     = "data/raw/reddit/raw_reddit_00/useryd.csv"

OUTPUT_DIR = "data/prepared/reddit/net_00"

# Need:
#	subreddit_id_2_countyid
#   countyid_2_fips
countyid_2_fips = {}
fips_set = set()
with open(COUNTY_FN, 'r') as county_f:
    county_f.readline()

    # "county_id";"county_name";"state";"fips"
    for line in county_f.readlines():
        raws = line.replace("\"", "").replace("\n", "").split(";")

        county_id = raws[0]
        fips = raws[3].zfill(5)  # TODO - LEFTPAD W 0's to LENGTH 5
        fips_set.add(fips)
        countyid_2_fips[county_id] = fips

subid_2_fips = {}
with open(SUBREDDIT_FN, 'r') as sub_f:
    sub_f.readline()

    # "subreddit_id";"subreddit_name";"subreddit_url";"county_id"
    for line in sub_f.readlines():
        raws = line.replace("\"", "").replace("\n", "").split(";")

        sub_id = raws[0]
        county_id = raws[3]

        if county_id == '0':
            continue
        else:
            fips = countyid_2_fips[county_id]
            subid_2_fips[sub_id] = fips

## User aggregation - Over all days in the year
# useryd.id -> User_reddit_id
userydid_2_redditid = {}
redditid_2_homesub = {}
with open(USERYD_FN, 'r') as user_f:
    user_f.readline()

    # "useryd_id";"user_reddit_id";"user_reddit_name";"year";"day";"home_subreddit"
    for line in user_f.readlines():
        raws = line.replace("\"", "").replace("\n", "").split(";")

        useryd_id = raws[0]
        reddit_id = raws[1]
        home_sub = raws[5]

        userydid_2_redditid[useryd_id] = reddit_id
        redditid_2_homesub[reddit_id] = home_sub

## User Activity Edges
# user_reddit_id -> user_edge_list (with counts) (sub_id, count)
# "subreddit_id";"useryd_id"
subreddits = set()
redditid_2_activity = {}
with open(ACTIVE_SUB_FN, 'r') as activity_f:
    activity_f.readline()

    for line in activity_f.readlines():
        raws = line.replace("\"", "").replace("\n", "").split(";")

        sub_id = raws[0]
        useryd_id = raws[1]

        subreddits.add(sub_id)

        if useryd_id in userydid_2_redditid:
            reddit_id = userydid_2_redditid[useryd_id]
        else:
            continue

        if reddit_id not in redditid_2_activity:
            redditid_2_activity[reddit_id] = {}

        activity_dict = redditid_2_activity[reddit_id]
        if sub_id not in activity_dict:
            activity_dict[sub_id] = 0
        activity_dict[sub_id] += 1

## User Localization (User-County Edges)
# user_reddit_id -> fips
redditid_2_fips = {}
for redditid in redditid_2_homesub:
    home_sub = redditid_2_homesub[redditid]

    if home_sub in subid_2_fips:
        redditid_2_fips[redditid] = subid_2_fips[home_sub]

## Goal is to have a 3 layer network: (User, Subreddit, County)
#  with only User--Subreddit, User--County, County--County edges.

# I think we have all the things we need to do that.
# What's the best way to turn this into a format that can be consumed by
# a thing like GraphSage?

# Need 2 tensors
# Nodes - (C+S+U, 3) - The list of counties, subs, users, with their one-hot vectors
# Edges - (|E|, 2) - The edge list, using the indexes from above.

### Nodes
nodes = []
idx = 0

# users
redditid_2_netid = {}
netid_2_redditid = {}
for ru in redditid_2_homesub:
    # print(ru, redditid_2_homesub[ru])
    redditid_2_netid[ru] = idx
    netid_2_redditid[idx] = ru
    nodes.append([1, 0, 0])  # the 'user' one-hot vector
    idx += 1

# subreddits
subid_2_netid = {}
netid_2_subid = {}
for sub in subreddits:
    subid_2_netid[sub] = idx
    netid_2_subid[idx] = sub
    nodes.append([0, 1, 0])  # subreddit one-hot
    idx += 1

# counties
fips_2_netid = {}
netid_2_fips = {}
for fips in fips_set:
    fips_2_netid[fips] = idx
    netid_2_fips[idx] = fips
    nodes.append([0, 0, 1])  # county one-hot
    idx += 1

### Edges
# TODO - Add in County-County edges
edges = []
# user-subreddits
for user_rid, activity_dict in redditid_2_activity.items():
    user_idx = redditid_2_netid[user_rid]

    for sub, count in activity_dict.items():
        sub_idx = subid_2_netid[sub]
        edges.append((user_idx, sub_idx))  # CHANGE HERE TO ADD WEIGHTS

# user-counties
for user_rid, fips in redditid_2_fips.items():
    user_idx = redditid_2_netid[user_rid]
    fips_idx = fips_2_netid[fips]
    edges.append((user_idx, fips_idx))  # CHANGE HERE TO ADD WEIGHTS


# We now have 6 maps to save, a node list, and an edge list.
def save_dict(dict_obj, fn):
    with open(fn, 'w') as f:
        for k, v in dict_obj.items():
            f.write("{}, {}\n".format(k, v))


def save_list(l, fn):
    with open(fn, 'w') as f:
        for row in l:
            row_str = [str(i) for i in row]
            f.write("{}\n".format(", ".join(row_str)))


save_dict(redditid_2_netid, "{}/rid_2_netid.csv".format(OUTPUT_DIR))
save_dict(netid_2_redditid, "{}/netid_2_rid.csv".format(OUTPUT_DIR))
save_dict(subid_2_netid, "{}/sid_2_netid.csv".format(OUTPUT_DIR))
save_dict(netid_2_subid, "{}/netid_2_sid.csv".format(OUTPUT_DIR))
save_dict(fips_2_netid, "{}/fips_2_netid.csv".format(OUTPUT_DIR))
save_dict(netid_2_fips, "{}/netid_2_fips.csv".format(OUTPUT_DIR))

save_list(nodes, "{}/nodes.csv".format(OUTPUT_DIR))
save_list(edges, "{}/edges.csv".format(OUTPUT_DIR))
