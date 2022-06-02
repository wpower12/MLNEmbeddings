import pandas as pd
import mysql.connector
from decouple import config
import pandas as pd

# DATABASE_NAME = "Full_Run_DB_00"  # Local.
DATABASE_NAME = "Full_Run_01" # For lab machine
cnx = mysql.connector.connect(user=config('DB_USER'),
                              password=config('DB_PASSWORD'),
                              host=config('DB_HOST'),
                              database=DATABASE_NAME)

DATA_DIR  = "../data/prepared/twitter"
FN_userid = "2022_01_01_to_07_userids_02.csv"
OUT_FN = "2022_01_01_to_07_userid_map_02.csv"

GET_FIPS = """
    SELECT id, mode_tweet_fips_00
    FROM user
    WHERE id={};
"""

userid_2_fips = {}
with open("{}/{}".format(DATA_DIR, FN_userid)) as f_ui:
    cur = cnx.cursor()
    for line in f_ui.readlines():
        u_id = line.replace("\n", "")
        cur.execute(GET_FIPS.format(u_id))
        userid_2_fips[u_id] = cur.fetchone()[1]

with open("{},{}".format(DATA_DIR, OUT_FN)) as f_out:
    for k in userid_2_fips:
        f_out.write("{},{}\n".format(k, userid_2_fips[k]))
