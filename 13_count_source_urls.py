import pandas as pd
import mysql.connector
from decouple import config

DATABASE_NAME = "Full_Run_01" # For lab machine
cnx = mysql.connector.connect(user=config('DB_USER'),
                              password=config('DB_PASSWORD'),
                              host=config('DB_HOST'),
                              database=DATABASE_NAME)

FN_SOURCES = "data/raw/sources.csv"
FN_OUT     = "data/raw/sources_w_counts.csv"
SQL_COUNT = """
    SELECT COUNT(*) FROM url
    WHERE url.url LIKE '%{}%';
"""

def make_count_from_db(cur):
    def count_from_db(url_stub):
        cur.execute(SQL_COUNT.format(url_stub))
        return cur.fetchone()[0]
    return count_from_db


def trim_url(url_str):
    return url_str.replace("http://", "").replace("https://", "").replace("www.", "").replace("/", "")


df = pd.read_csv(FN_SOURCES, header=0)
df['url_stub'] = df['URL'].apply(trim_url)
df['tweet_count'] = df['url_stub'].apply(make_count_from_db(cnx.cursor()))
df.to_csv(FN_OUT, index=False)
