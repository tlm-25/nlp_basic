from src.helpers import utils 
#name of column with the text description of an incident
TEXT_COLUMN_NAME = 'Report'
#column name with the output label
LABEL_COLUMN = 'Occurence Nature condition'

# number of topics that we are interested in 
NUM_TOPICS = 6

#Extract the N most commin words per topic
NUM_WORDS = 8

###READ IN DATA

#input file name
FILE_NAME = 'airline_incidents.csv'

##RANDOM SEED GENERATOR
RANDOM_STATE = 42


#import data
incidents_df = utils.import_data(FILE_NAME)
print(incidents_df.info())
print(incidents_df[LABEL_COLUMN].value_counts())



#Get stratified sample of the dataset
stratified_incidents = utils.stratify_data(df=incidents_df,proportion=0.1,output_column=LABEL_COLUMN,random_seed=RANDOM_STATE)
print(stratified_incidents.info())
print(stratified_incidents[LABEL_COLUMN].value_counts())


# #clean the text data (lowercase, tokenize etc.)
# incidents_df = utils.prepare_text(incidents_df,TEXT_COLUMN_NAME)



# #get the column with the incident descriptions 
# descriptions = incidents_df[TEXT_COLUMN_NAME]