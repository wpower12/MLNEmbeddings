import pandas as pd

DATA_DIR  = "../data/prepared/twitter"
FN_LEMMAS = "2022_01_01_to_07_lemmas_02.csv"
OUT_FN = "2022_01_01_to_07_userids_02.csv"

df = pd.read_csv("{}/{}".format(DATA_DIR, FN_LEMMAS), header=0, index_col=0)

user_ids = df['userid'].unique()
print(user_ids)

with open("{}/{}".format(DATA_DIR, OUT_FN), "w") as f:
    for u in user_ids:
        f.write("{}\n".format(u))
