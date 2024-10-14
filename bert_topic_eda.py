
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic
from src.helpers import utils
from src.helpers import vis
import pandas as pd

#name of column with the text description of an incident
TEXT_COLUMN_NAME = 'description'
#column name with the output label
LABEL_COLUMN = 'type'

# number of topics that we are interested in 
NUM_TOPICS = 6

#Extract the N most commin words per topic
NUM_WORDS = 8

###READ IN DATA

#input file name
FILE_NAME = 'dummy_data.csv'

#import data
incidents_df = utils.import_data(FILE_NAME)


#clean the text data (lowercase, tokenize etc.)
incidents_df = utils.prepare_text(incidents_df,TEXT_COLUMN_NAME)

#get the column with the incident descriptions 
descriptions = incidents_df[TEXT_COLUMN_NAME].astype(str)


topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

topics, probs = topic_model.fit_transform(descriptions)

topics_df = pd.DataFrame({"Document": descriptions, "Topic:": topics})

print(topics_df.head(15))


#topics_df.loc[topics_df["Topics"]== 0,"Topic"] == "theft"
#topics_df.loc[topics_df["Topics"]==1,"Topic"] == ""
#topics_df.loc[topics_df["Topics"]==2,"Topic"] == ""
#topic_model.visualize_barchart()