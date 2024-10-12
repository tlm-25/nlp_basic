
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.helpers import utils
from src.helpers import vis

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



RANDOM_STATE = 42

#import data
incidents_df = utils.import_data(FILE_NAME)

#clean the text data (lowercase, tokenize etc.)
incidents_df = utils.prepare_text(incidents_df,TEXT_COLUMN_NAME)

#get the column with the incident descriptions 
descriptions = incidents_df[TEXT_COLUMN_NAME]

## EXPLORATORY ANALYSIS

##LOOK AT DISTRIBUTION OF LABELS IN THE DATASET


##TOPIC MODELLING

#get all unique words in dataset
dictionary = gensim.corpora.Dictionary(descriptions)

#create bag of words from the descriptions
doc_term = [dictionary.doc2bow(text) for text in descriptions ]

#generate topics 
lda_topics = gensim.models.LdaModel(corpus=doc_term,id2word=dictionary,num_topics=NUM_TOPICS,random_state=RANDOM_STATE)
#get common words for each topic and produce plots 
topic_words = lda_topics.print_topics(num_topics=NUM_TOPICS,num_words=NUM_WORDS)
print(topic_words)
vis.topic_word_cloud(lda_topics)

vis.view_label_sample(incidents_df,LABEL_COLUMN)

l=[lda_topics.get_document_topics(item) for item in data1]

#Implement BERT Topic to see how topics are related, and see how many topics would be useful
#see the interaction between different clusters. 
