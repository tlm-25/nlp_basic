from src.helpers import utils, label_clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
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

### PROPORTION FOR STRATIFIED SAMPLE
SAMPLE_PROPORTION = 0.25

### MINIMUM NUMBER OF DATA POINTS FOR A CATEGORY IN THE STRATIFIED SAMPLE
MIN_STRAT_SAMPLES = 100


#import data
incidents_df = utils.import_data(FILE_NAME)
#print(incidents_df.info())
#print(incidents_df[LABEL_COLUMN].value_counts())



#Get stratified sample of the dataset - reduce to 25% of size of original dataset (reduce amount of manual verification)
stratified_incidents = utils.stratify_data(df=incidents_df,proportion=SAMPLE_PROPORTION,output_column=LABEL_COLUMN,random_seed=RANDOM_STATE)
print(stratified_incidents.info())
#print(stratified_incidents[LABEL_COLUMN].value_counts())


#Use a random forest to make predictions on the labels - random forest has bootstrapping integrated into it


#Prepare the text (lemmatize, tokenize, remove stopwords etc. - create a bag of words)
# deliberately kept simple to avoid overfitting - i.e. prevent agreeing "too much" with the original dataset in case the dataset has mislabels in it - only flagging areas of potential disagreement 
stratified_incidents = utils.prepare_text(stratified_incidents,TEXT_COLUMN_NAME)


stratified_incidents[TEXT_COLUMN_NAME] = stratified_incidents[TEXT_COLUMN_NAME].astype(str)
#input column(s) (x)
incident_descriptions = stratified_incidents[TEXT_COLUMN_NAME]
#output colum (y)
incident_categories = stratified_incidents[LABEL_COLUMN]



#Doing simple train test set, not doing validation set at this stage - don't want to tune TOO closely to the data in
#case there are many bad laebles 

#incident_descriptions_bow = bow_generator.fit_transform(stratified_incidents[TEXT_COLUMN_NAME])
#TF-IDF transformer  on the text test
bow_tfidf= TfidfVectorizer(ngram_range=(1,2),min_df=5, max_df=0.75)
incident_descriptions_tfidf = bow_tfidf.fit_transform(incident_descriptions)
x_train,x_test,y_train,y_test = train_test_split(incident_descriptions_tfidf,incident_categories,test_size=0.2,random_state=RANDOM_STATE,stratify=incident_categories)
test_descriptions = train_test_split(incident_descriptions,incident_categories,test_size=0.2,random_state=RANDOM_STATE,stratify=incident_categories)[1]
#print(incident_descriptions_tfidf)
min_samples = y_train.value_counts().min()
oversample = SMOTE(random_state=RANDOM_STATE,k_neighbors=min_samples-1)
x_res,y_res = oversample.fit_resample(x_train,y_train)

#ros = RandomOverSampler(random_state=RANDOM_STATE)
model = rf(max_depth=7,n_estimators=200,random_state=RANDOM_STATE)
model.fit(x_res,y_res)

predictions =  model.predict(x_test)

#print(x_res.info())

headers = ["description","label","prediction"]

predictions_df = pd.DataFrame({"description": test_descriptions, "label":y_test,"prediction":predictions})


print("test f1 score: ",f1_score(y_test,predictions,average="weighted"))

#Get the points where the model predicts differently to the actual dataset 
label_disagree_df = label_clean.get_label_disagreements(predictions_df,"label","prediction")

label_disagree_csv = label_disagree_df.to_csv("doubtful_label.csv")






