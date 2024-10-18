import pandas as pd
import csv 

##FUNCTION TO COMPARE PREDICTIONS AND RETURN A DATAFRAME WITH THE LABEL DISAGREEMENTS

def get_label_disagreements(df:pd.DataFrame,test_label_column:str, test_prediction_column:str):

    if test_label_column not in df.columns:     
        raise IndexError(f"The column {test_label_column} is not in the dataframe. Please ensure it is spelt correctly")
    
    if test_prediction_column not in df.columns:     
        raise IndexError(f"The column {test_prediction_column} is not in the dataframe. Please ensure it is spelt correctly") 
    
    label_disagree_df = df.loc[df[test_label_column]!=df[test_prediction_column]]

    return label_disagree_df
    



#replace values in original df with vlaues changed in csv if file change detected 


# headers = ["description","label","prediction"]

# predictions_df = pd.DataFrame({"description": test_descriptions, "label":y_test,"prediction":predictions})
# #predictions_csv = predictions_df.to_csv("predictions.csv")

# # #clean the text data (lowercase, tokenize etc.)

# print("test f1 score: ",f1_score(y_test,predictions,average="weighted"))


# label_disagree_df = predictions_df.loc[predictions_df["label"]!=predictions_df["prediction"]]

# label_disagree_csv = label_disagree_df.to_csv("doubtful_label.csv")


##AFTER THIS

#create a re_train_with_new_labels function 

#correct the doubtful labels, then replace the values in the original dataset (replace on id)
#only run the correction function if the predictions csv file exists
#retrain


#consider cleanlab approach 




