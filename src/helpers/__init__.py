import pandas as pd 
import os 

input_file = 'incidents.csv'

def import_data():
    if not os.path.exists(input_file):
        print(f"Error: {input} could not be found")
    
    try:
        input_df = pd.read_csv(input_file)

    except Exception as error:
        print(f"There was a problem reading the file: {error}")





