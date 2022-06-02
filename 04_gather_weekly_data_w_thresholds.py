import mysql.connector
from decouple import config
import pandas as pd

# DATABASE_NAME = "Full_Run_DB_00"  # Local.
DATABASE_NAME = "Full_Run_01" # For lab machine
cnx = mysql.connector.connect(user=config('DB_USER'),
                              password=config('DB_PASSWORD'),
                              host=config('DB_HOST'),
                              database=DATABASE_NAME)

FN_STUB = "2022_01_01_to_07_thresh_{}.csv"
DATA_DIR = "data/raw/twitter/thresholded"

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
START_DATE = '2022-01-01'
END_DATE = '2022-01-07'
COL_NAME = 'dupe_ratio_05'  # TODO - Check this.

# Format parameters: DATE_START, DATE_END, DUPE_RATIO_COL_NAME, THRESHOLD
GET_TWEETS = """
	SELECT userid, text 
	FROM tweet
	LEFT JOIN user on user.id=tweet.userid
	WHERE 
		tweet.created_at BETWEEN '{}' AND '{}'
	AND
		user.{} <= {};
"""

# This should do it. I think there might be an issue with saving out the csv. Might have to change to a
# different deliminator? How does mysql workbench handle this gracefully?
for t in THRESHOLDS:
	# cur = cnx.cursor()
	# cur.execute(GET_TWEETS.format(START_DATE, END_DATE, COL_NAME, t))
	
	# tweets = list(cur.fetchall())
	# t_df = pandas.DataFrame(tweets, columns=["userid", "text"])
	t_df = pd.read_sql(GET_TWEETS.format(START_DATE, END_DATE, COL_NAME, t), cnx)
	t_df.to_csv(FN_STUB.format(t))


	print("wrote: {}".format( FN_STUB.format(t)))

