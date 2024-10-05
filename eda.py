
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.helpers import utils

#input file
FILE_NAME = 'dummy_data.csv'

#name of the column where the user enters freetext to describe the incident 
INCIDENT_DESC_COL = 'description'

#get the number of topics that we are interested in 
num_topics = 6

#import data
incidents_df = utils.import_data(FILE_NAME)

#clean the text data (lowercase, tokenize etc.)
incidents_df = utils.prepare_text(incidents_df,incident_description_column)

descriptions = incidents_df[incident_description_column]


###TOPIC MODELLING

#get unique words in dataset
dictionary = gensim.corpora.Dictionary(descriptions)

print(dictionary)

#create bag of words from the descriptions
doc_term = [dictionary.doc2bow(text) for text in descriptions ]
#print(doc_term)

lda_model = gensim.models.LdaModel(corpus=doc_term,id2word=dictionary,num_topics=6)


topic_words = lda_model.print_topics(num_topics=num_topics,num_words=8)

#print(type(topic_words[0]))


for i in range(0,len(topic_words)):
     word_cloud = WordCloud(background_color='black', width=800,height=500,random_state=21,max_font_size=110).generate_from_frequencies(topic_words[i])
     plt.figure(figsize=(10,7))
     plt.imshow(word_cloud,interpolation='bilinear')
     plt.axis("off")
     plt.title("topic"+ " "+ str(i))
     plt.show()
