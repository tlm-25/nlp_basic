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

en_stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def import_data(input_file):
    
    try:
        input_df = pd.read_csv(input_file,encoding='utf-8')

        return input_df

    except FileNotFoundError:
        raise FileNotFoundError(f"'{input_file}' could not be found. Please check the directory and ensure that it is spelt correctly ")



def prepare_text(df,column_name):
    #make sure that the user enters function arguments correctly. If not, raise an error
    if not isinstance(column_name,str) :
        raise TypeError(f"Please enter a string in the second argument of the clean_text function: clean_text(df,'<column_name>')")
    
    
    ###DATAFRAME READING BLOCK
    
    #check that the column exists, otherwise raise an error 
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in the dataframe. Please check columns and ensure it correctly spelt")
    
    #check that the type of the column is a string dtype. If not, raise a TypeError. pd.api.types can check 'objects' and newer StringDtypes which makes it robust
    if not pd.api.types.is_string_dtype(df[column_name]):
        raise TypeError(f"Please choose a column of  type <string>. The column, '{column_name}' is of type <{df[column_name].dtype}>")
    
    ####TEXT PREP BLOCK
    try:

        #convert to lowercase
        df[column_name] = df[column_name].str.lower()
        
        #remove stopwords - common words that don't add much meaning (e.g. and, of, a, to)
        df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in  x.split() if word not in en_stopwords]))

        #remove punctuation - regex pattern 
        #anything that is NOT whitespspace (\s) or a word (\w) is considered punctuation
        punctuation = r"[^\w\s]"

        #remove punctuation by subbbing punctuation for nothing
        df[column_name] = df[column_name].apply(lambda x: re.sub(punctuation,'',x))

        

        #lemmatize text - e.g. breaking to break, playing becomes play
        df[column_name] = df[column_name].apply(lambda x: lemmatizer.lemmatize(x))

        #tokenise - break up sentences into individuual words
        df[column_name] = df[column_name].apply(lambda x: word_tokenize(x))



        print("Finished cleaning text. You can proceed to train your model")
        return df 

    except Exception as e:
        print(f"Error processing text data: {e}")
        print("Please re-look  at the 'text processing section' your clean_text function")




