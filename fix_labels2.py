from cleanlab import Datalab
from cleanlab.classification import CleanLearning
import pandas as pd
from src.helpers import utils
from sklearn.ensemble import RandomForestClassifier as rf

#name of column with the text description of an incident
TEXT_COLUMN_NAME = 'Report'

#column name with the output label
LABEL_COLUMN = 'Occurence Nature condition'

###READ IN DATA

#input file name
FILE_NAME = 'airline_incidents.csv'

##RANDOM SEED GENERATOR
RANDOM_STATE = 42

### PROPORTION FOR STRATIFIED SAMPLE
SAMPLE_PROPORTION = 0.25

### MINIMUM NUMBER OF DATA POINTS FOR A CATEGORY IN THE STRATIFIED SAMPLE
MIN_STRAT_SAMPLES = 100

incidents_df = utils.import_data(FILE_NAME)
#print(incidents_df.info())
#print(incidents_df[LABEL_COLUMN].value_counts())



# #Get stratified sample of the dataset - reduce to 25% of size of original dataset (reduce amount of manual verification)
# stratified_incidents = utils.stratify_data(df=incidents_df,proportion=SAMPLE_PROPORTION,output_column=LABEL_COLUMN,random_seed=RANDOM_STATE)

# lab = Datalab(data=stratified_incidents,label_name=LABEL_COLUMN)
# #lab.find_issues(features=feature_embeddings,pred_probs=pred_probs)

# label_issues_info = CleanLearning(clf=rf).find_label_issues(stratified_incidents,stratified_incidents[LABEL_COLUMN])

# lab.report()