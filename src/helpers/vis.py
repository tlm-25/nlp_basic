import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd 
import seaborn as sns

#generate word cloud for the labels in the dataset
def topic_word_cloud(lda_model):

    '''for each topic in the number of topics generated '''
    for topic in range(lda_model.num_topics):

        plt.figure()

        '''plot a word cloud for each topic (i.e. view common words per topic)'''
        plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(topic,200))))
        plt.axis("off")
        plt.title("Topic "+ " "+ str(topic))
        plt.show()

#view sample of labels in the dataset 
def view_label_sample(df:pd.DataFrame,column_name:str):
    label_split = df[column_name].value_counts().reset_index()
    print(label_split.info())
    label_split.rename(columns ={"index":"type"},inplace=True)
    label_split.rename(columns ={f"{column_name}":"count"},inplace=True)
    plt.figure()
    sns.barplot(y=label_split["type"],x=label_split["count"])
    plt.show()
    
    # print(label_split)
    # word frequency analysis 
    # average word legnth analysis 
    # sentence length analysis 
    #
    


