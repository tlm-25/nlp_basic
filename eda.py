from src.helpers import utils
import gensim

file_name = 'dummy_data.csv'
incident_description_column = 'description'

#import data
incidents_df = utils.import_data(file_name)

#clean the text data (lowercase, tokenize etc.)
incidents_df = utils.prepare_text(incidents_df,incident_description_column)

#print(incidents_df[incident_description_column].head(9))

###TOPIC MODELLING
#get unique words in dataset
dictionary = gensim.corpora.Dictionary(incidents_df[incident_description_column])

#beg of words
doc_term
