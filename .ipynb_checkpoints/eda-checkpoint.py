from src.helpers import utils
import gensim

file_name = 'dummy_data.csv'
incident_description_column = 'description'

#get the number of topics that we are interested in 
num_topics = 6

#import data
incidents_df = utils.import_data(file_name)

#clean the text data (lowercase, tokenize etc.)
incidents_df = utils.prepare_text(incidents_df,incident_description_column)

descriptions = incidents_df[incident_description_column]


###TOPIC MODELLING

#get unique words in dataset
dictionary = gensim.corpora.Dictionary(descriptions)

print(dictionary)

#create bag of words from the descriptions
doc_term = [dictionary.doc2bow(text) for text in descriptions ]
print(doc_term)



lda_model = gensim.models.LdaModel(corpus=doc_term,id2word=dictionary,num_topics=6)

print(lda_model.print_topics(num_topics=num_topics,num_words=5))

#display the topic groups 