import pandas as pd 
import os 
import nltk 
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import sklearn as sk
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

en_stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def import_data(input_file):
    
    try:
        input_df = pd.read_csv(input_file,encoding='utf-8')

        return input_df

    except FileNotFoundError:
        raise FileNotFoundError(f"'{input_file}' could not be found. Please check the directory and ensure that it is spelt correctly ")
    


## TEXT PRE-PROCESSING
def prepare_text(df,column_name:str):

    '''make sure that the user enters function arguments correctly. If not, raise an error'''
    if not isinstance(column_name,str) :
        raise TypeError(f"Please enter a string in the second argument of the clean_text function: clean_text(df,'<column_name>')")
    
    
    ###DATAFRAME READING BLOCK
    
    '''check that the column exists. raise a KeyError if not ''' 

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in the dataframe. Please check columns and ensure it correctly spelt")
    
    '''check that the data type of the column is <string> dtype. If not, raise a TypeError. pd.api.types can check 'objects' and the newer StringDtypes which makes it robust'''

    if not pd.api.types.is_string_dtype(df[column_name]):
        raise TypeError(f"Please choose a column of  type <string>. The column, '{column_name}' is of type <{df[column_name].dtype}>")
    
    ####TEXT PREP BLOCK
    try:

        '''convert to lowercase'''
        df[column_name] = df[column_name].str.lower()
        
        '''remove stopwords - common words that don't add much meaning (e.g. and, of, a, to)'''
        df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in  x.split() if word not in en_stopwords]))

        '''remove punctuation - anything that is not whitespspace (\s) or a word (\w) is considered punctuation'''
        punctuation = r"[^\w\s]"
        df[column_name] = df[column_name].apply(lambda x: re.sub(punctuation,'',x))

        

        '''lemmatize text - e.g. breaking to break, playing becomes play'''
        df[column_name] = df[column_name].apply(lambda x: lemmatizer.lemmatize(x))

        '''tokenise - break up sentences into individuual words'''
        df[column_name] = df[column_name].apply(lambda x: word_tokenize(x))



        print("Finished cleaning text. You can proceed to train your model")
        return df 

    except Exception as e:
        print(f"Error processing text data: {e}")
        print("Please re-look  at the 'text processing section' your clean_text function")



###TAKE A STRATIFIED SAMPLE FROM DATAFRAME
def stratify_data(df:pd.DataFrame,output_column:str,proportion:float,random_seed:int):
    
    '''If proportion argument entered is greater than 1, raise an error'''
    if proportion > 1: 
        raise ValueError("proportion argument must be between 0 and 1 e.g. 0.4 for 40% of dataset")
    
    '''get expected proportion of each label in the dataset'''
    df_stratified = df.groupby(output_column).apply(lambda x: x.sample(frac=proportion,random_state=random_seed))
    
    return df_stratified














# ###CORRECTING LABELS

# def correct_labels(df:pd.DataFrame,output_file_name:str):

#     stratified_df = df

#     n_iteration = 0 
#     #record_time_file_generated
#     while os.path.exists(f"{}")
    
#     while os.path.exists('./pre_predictions_{}.csv'):
#         raise War

#check if file already exist 
#if os.path.exists(./predictions.csv)
#raise Warning("predictions.csv already exists.)
#while:
# user_input = input("Do you want to continue and overwrite it? Enter "Y" if yes, "N" No.")
# if user_input!="Y" and user_input not equal to no 

#if file already exists, raise a warning and confirm with the user if they'd like to overwrite the exisiting "predictions file"

#stratify the sample dataset

#clean labels and output a new dataset 




#bootstrap the model and find the best performing model

#output a csv file and manually evaluate the labels

#repeat this function until satisfactoryt
