import matplotlib.pyplot as plt
from wordcloud import WordCloud


def topic_word_cloud(lda_model):
    for topic in range(lda_model.num_topics):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(topic,200))))
        plt.axis("off")
        plt.title("Topic "+ " "+ str(topic))
        plt.show()


