from mlnhelper import utils


def build_twitter_mln_from_raws(fn_userid2fips, fn_bowid2term, fn_user_agged_bows, fn_county_adj=None):
    nodes = []
    counties = set()
    user_2_fips = {}
    with open(fn_userid2fips) as f_userid_2_fips:
        for line in f_userid_2_fips:
            userid, fips = line.replace("\n", "").split(",")
            user_2_fips[userid] = fips
            counties.add(fips)

    bowid_2_term = {}
    with open(fn_bowid2term) as f_bow_2_term:
        for line in f_bow_2_term:
            term, bowid = line.replace("\n", "").split(", ")
            bowid_2_term[bowid] = term

    # Use the two maps derived above to generate the consistent node indexes, while we also create the feature vectors.
    fips_2_idx = {}
    fips_n_idx = 0
    for county in counties:
        fips_2_idx[county] = fips_n_idx
        nodes.append([1, 0, 0])
        fips_n_idx += 1

    user_2_idx = {}
    user_n_idx = fips_n_idx + 1
    for user in user_2_fips:
        user_2_idx[user] = user_n_idx
        nodes.append([0, 1, 0])
        user_n_idx += 1

    bowid_2_idx = {}
    bowid_n_idx = user_n_idx + 1
    for bowid in bowid_2_term:
        bowid_2_idx[bowid] = bowid_n_idx
        nodes.append([0, 0, 1])
        bowid_n_idx += 1

    # Edges -
    #   User-County - Using the user_2_fips dict.
    #   User-Term   - Using an aggregate of the bows?
    #   County-County - Need the fips adjacency list.
    edges = []
    for user in user_2_fips:
        fips = user_2_fips[user]
        user_n_idx = user_2_idx[user]
        fips_n_idx = fips_2_idx[fips]

        if user_n_idx < len(nodes) and fips_n_idx < len(nodes):
            edges.append((user_n_idx, fips_n_idx))
            edges.append((fips_n_idx, user_n_idx))

    if fn_county_adj is not None:
        cadj_map = utils.cadj_from_txt(fn_county_adj)
        for county_from in cadj_map:
            if county_from in fips_2_idx:
                node_idx_from = fips_2_idx[county_from]
                counties_to = cadj_map[county_from]
                for fips in counties_to:
                    if fips in fips_2_idx:
                        node_idx_to = fips_2_idx[fips]
                        edges.append((node_idx_from, node_idx_to))
                        edges.append((node_idx_to, node_idx_from))

    # Need to load the agged_bows (aggregated bag-of-words features)
    user_2_bows = {}
    with open(fn_user_agged_bows) as f_bows:
        f_bows.readline()
        for line in f_bows.readlines():
            userid, bow_str = line.replace("\n", "").split(",", 1)
            user_2_bows[userid] = utils.str_to_dict(bow_str)

    for user in user_2_bows:
        bow_dict = user_2_bows[user]
        user_n_idx = user_2_idx[user]

        for bow_id in bow_dict:
            # weight = bow_dict[bow_id]
            bowid_n_idx = bowid_2_idx[bow_id]
            if user_n_idx < len(nodes) and bowid_n_idx < len(nodes):
                edges.append((user_n_idx, bowid_n_idx))
                edges.append((bowid_n_idx, user_n_idx))

    return nodes, edges, {'fips': fips_2_idx, 'userid': user_2_idx, 'bowid': bowid_2_idx}


def build_twitter_mln_from_fns(fn_userid2fips, fn_bowid2term, fn_user_agged_bows, fn_county_adj=None):
    counties = set()
    user_2_fips = {}
    with open(fn_userid2fips) as f_userid_2_fips:
        for line in f_userid_2_fips:
            userid, fips = line.replace("\n", "").split(",")
            user_2_fips[userid] = fips
            counties.add(fips)

    bowid_2_term = {}
    with open(fn_bowid2term) as f_bow_2_term:
        for line in f_bow_2_term:
            term, bowid = line.replace("\n", "").split(", ")
            bowid_2_term[bowid] = term

    user_2_bows = {}
    with open(fn_user_agged_bows) as f_bows:
        f_bows.readline()
        for line in f_bows.readlines():
            userid, bow_str = line.replace("\n", "").split(",", 1)
            user_2_bows[userid] = utils.str_to_dict(bow_str)

    if fn_county_adj is not None:
        cadj = utils.cadj_from_txt(fn_county_adj)
        return build_twitter_mln_from_maps(user_2_fips, bowid_2_term, user_2_bows, counties, cadj_map=cadj)
    return build_twitter_mln_from_maps(user_2_fips, bowid_2_term, user_2_bows, counties)


def build_twitter_mln_from_maps(user_2_fips, bowid_2_term, user_2_bows, counties, cadj_map=None):

    nodes = []
    fips_2_idx = {}
    fips_n_idx = 0
    for county in counties:
        fips_2_idx[county] = fips_n_idx
        nodes.append([1, 0, 0])
        fips_n_idx += 1

    user_2_idx = {}
    user_n_idx = fips_n_idx + 1
    for user in user_2_fips:
        user_2_idx[user] = user_n_idx
        nodes.append([0, 1, 0])
        user_n_idx += 1

    bowid_2_idx = {}
    bowid_n_idx = user_n_idx + 1
    for bowid in bowid_2_term:
        bowid_2_idx[bowid] = bowid_n_idx
        nodes.append([0, 0, 1])
        bowid_n_idx += 1

    # Edges -
    #   User-County - Using the user_2_fips dict.
    #   User-Term   - Using an aggregate of the bows?
    #   County-County - Need the fips adjacency list.
    edges = []
    for user in user_2_fips:
        fips = user_2_fips[user]
        user_n_idx = user_2_idx[user]
        fips_n_idx = fips_2_idx[fips]

        if user_n_idx < len(nodes) and fips_n_idx < len(nodes):
            edges.append((user_n_idx, fips_n_idx))
            edges.append((fips_n_idx, user_n_idx))

    if cadj_map is not None:
        for county_from in cadj_map:
            if county_from in fips_2_idx:
                node_idx_from = fips_2_idx[county_from]
                counties_to = cadj_map[county_from]
                for fips in counties_to:
                    if fips in fips_2_idx:
                        node_idx_to = fips_2_idx[fips]
                        edges.append((node_idx_from, node_idx_to))
                        edges.append((node_idx_to, node_idx_from))

    for user in user_2_bows:
        bow_dict = user_2_bows[user]

        if user not in user_2_idx:
            continue
        user_n_idx = user_2_idx[user]

        for bow_id in bow_dict:
            # weight = bow_dict[bow_id]
            bowid_n_idx = bowid_2_idx[bow_id]
            if user_n_idx < len(nodes) and bowid_n_idx < len(nodes):
                edges.append((user_n_idx, bowid_n_idx))
                edges.append((bowid_n_idx, user_n_idx))

    return nodes, edges, {'fips': fips_2_idx, 'userid': user_2_idx, 'bowid': bowid_2_idx}


def aggregate_user_tweets(df):
    # assuming df has columns [idx, tweet-id, lemmas, bow] and that the bow is already a list, not a string.
    user_2_bows = {}
    for index, row in df.iterrows():
        bow_list = row['bow']
        userid = row['userid']

        if userid not in user_2_bows:
            user_2_bows[userid] = {}

        # Just going to sum the bows for now.
        user_bow_dict = user_2_bows[userid]
        for entry in bow_list:
            bow_id, _ = entry
            if bow_id not in user_bow_dict:
                user_bow_dict[bow_id] = 1
            else:
                user_bow_dict[bow_id] += 1
    return user_2_bows
