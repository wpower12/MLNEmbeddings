import pandas as pd
from . import utils, preprocessing, networks, embedding, viz


def twitter_mln_embedding_viz_pipeline(fn_raw, fn_topics, userid_2_fips, counties, fn_viz, stopwords=None, cadj_map=None):
    df_raw = pd.read_csv(fn_raw,
                         index_col=0,
                         header=0,
                         dtype={'': int, 'userid': str, 'text': str},
                         low_memory=False)
    print("raw df loaded.")

    topic_terms = utils.read_topic_terms(fn_topics, stopwords=stopwords)
    print("terms loaded.")

    df_filtered, bow_term_dict = preprocessing.filter_lemmas(df_raw, topic_terms)  # df_filtered [id, lemmas, bow]
    print("lemmas filtered, aggregating user tweets.")

    user_2_agged_bows = networks.aggregate_user_tweets(df_filtered)
    print("user tweets aggregated, building MLN.")

    # make MLN
    nodes, edges, maps = networks.build_twitter_mln_from_maps(userid_2_fips,
                                                              bow_term_dict,
                                                              user_2_agged_bows,
                                                              counties,
                                                              cadj_map=cadj_map)
    print("MLN generated with |N| = {}, |E| = {}".format(len(nodes), len(edges)))

    print("generating embeddings and low dim representations.")
    embeddings, node_labels = embedding.learn_embeddings(nodes, edges)
    low_dim_rep_df = embedding.tsne(embeddings)
    low_dim_rep_df['label'] = node_labels

    print("generating embedding viz")
    viz.save_embedding_viz(low_dim_rep_df, fn_viz)



