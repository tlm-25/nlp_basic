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
def prepare_text(df:pd.DataFrame,column_name:str):

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

#FEEDBACK - More subtle errors - what if one of the strings is empty, or has an empty space 
#if column.isnull():
#raise warning 
#add logging 
# expert - in the real-world scenario 
#data expectations 
#handle all strange edge cases for a working product to get the 'expert' grade. 
