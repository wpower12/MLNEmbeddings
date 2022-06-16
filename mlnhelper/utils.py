import string
from nltk.corpus import stopwords


def clean_urls(dframe):
    dframe['text'] = dframe['text'].str.replace(r"http\S+", "", regex=True)


def make_preprocess(tkz, ltz, stopwords):
    def preprocess(text):
        if text is not None:
            tokens = tkz.tokenize(text)
            tokens = [t.lower() for t in tokens if t not in stopwords]
            lemmas = [ltz.lemmatize(t) for t in tokens]
            return lemmas
        return []
    return preprocess


def str_to_tuplelist(s):
    if s == "[]":
        return []
    tuples = s.replace("[", "").replace("]", "").replace("), (", "):(").split(":")
    tuples = [s.replace("(", "").replace(")", "") for s in tuples]
    tuples = [s.split(", ") for s in tuples]
    tuples = [(int(s[0]), int(s[1])) for s in tuples]
    return tuples


def str_to_dict(s):
    if s == "{}": return {}
    entries = s.replace("{", "").replace("}", "").split(", ")
    ret_dict = {}
    for entry in entries:
        bowid, weight = entry.split(":")
        ret_dict[bowid] = int(weight)
    return ret_dict


def read_text_dict(fn):
    text_dict = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            raw = line.replace("\n", "", ).split(", ")
            if len(raw) == 2:
                k, v = raw
                text_dict[int(v)] = k  # Because we need id -> string
    return text_dict


def make_clean_lemmas(topic_terms):
    def clean_lemmas(text):
        raw_lemmas = text.replace("]", "").replace("[", "").split(", ")
        good_lemmas = []
        for l in raw_lemmas:
            l = l.replace("'", "")
            if l in topic_terms and len(l) > 0:
                good_lemmas.append(l)
        return good_lemmas
    return clean_lemmas


def anx_dict_from_df(df):
    ret_dict = {}
    for _, row in df.iterrows():
        ret_dict[row['lemma']] = row['anxiety']
    return ret_dict


def cadj_from_txt(fn):
    county_adj_map = {}
    with open(fn, "r") as f:
        from_county = None
        for line in f.readlines():
            line = line.replace('\n', "").split("\t")
            if line[0] != '':
                # we have a new 'from county'
                from_county = line[1]
                # theres always a first adj on the same line?
                county_adj_map[from_county] = [line[3]]
            else:
                # we're adding to a 'current' from_county
                county_adj_map[from_county].append(line[3])
    return county_adj_map


def read_topic_terms(fn, stopwords=None):
    terms = set()
    with open(fn, "r") as f_topics:
        for topic in f_topics.readlines():
            term_str = topic.split(", ")[1]
            topic_terms = term_str.split(" + ")
            topic_terms = [s.replace("\n", "") for s in topic_terms]
            topic_terms = [s.split("*")[1][1:-1].replace("'", "") for s in topic_terms]
            for t in topic_terms:
                if stopwords is not None:
                    if t not in stopwords:
                        terms.add(t)
                else:
                    terms.add(t)
    return terms


def load_augmented_stopwords():
    stopwords_eng = stopwords.words("english")
    stopwords_eng += string.punctuation
    stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    stopwords_eng += [', ', "’, ’", '’, ’', '’, ’', '\'', "'", "`", "’"]
    return stopwords_eng


def read_user_map(fn):
    counties = set()
    user_2_fips = {}
    with open(fn) as f_userid_2_fips:
        for line in f_userid_2_fips:
            userid, fips = line.replace("\n", "").split(",")
            user_2_fips[userid] = fips
            counties.add(fips)
    return user_2_fips, counties
