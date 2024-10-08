
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.helpers import utils
from src.helpers import vis

#input file
FILE_NAME = 'dummy_data.csv'

#name of column with the text description of an incident
TEXT_COLUMN_NAME = 'description'

# number of topics that we are interested in 
NUM_TOPICS = 6

#Extract the N most commin words per topic
NUM_WORDS = 8

#import data
incidents_df = utils.import_data(FILE_NAME)

#clean the text data (lowercase, tokenize etc.)
incidents_df = utils.prepare_text(incidents_df,TEXT_COLUMN_NAME)

#get the column with the incident descriptions 
descriptions = incidents_df[TEXT_COLUMN_NAME]


###TOPIC MODELLING

#get unique words in dataset
dictionary = gensim.corpora.Dictionary(descriptions)

print(dictionary)

#create bag of words from the descriptions
doc_term = [dictionary.doc2bow(text) for text in descriptions ]

lda_topics = gensim.models.LdaModel(corpus=doc_term,id2word=dictionary,num_topics=NUM_TOPICS)

topic_words = lda_topics.print_topics(num_topics=NUM_TOPICS,num_words=NUM_WORDS)

#generate word cloud for the important topics 
vis.topic_word_cloud(lda_topics)


#Implement BERT Topic to see how topics are related, and see how many topics would be useful
#see the interaction between different clusters. 
