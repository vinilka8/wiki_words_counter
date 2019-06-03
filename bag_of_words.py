# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:59:12 2019

@author: Johmnn
"""
import logging
import pandas as pd
import urllib.request, json
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)


def get_data_from_url(page_id):
    """
    This function gets data from wiki url page
    """
    URL = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids=' + str(
        page_id) + '&explaintext&format=json'
    with urllib.request.urlopen(URL) as url:
        return json.loads(url.read().decode())


def get_title(web_data, page_id):
    """
    This function gets the titel of the web page
    """
    return web_data['query']['pages'][str(page_id)]['title']


def word_cleaning_and_extraction(sentence):
    """
    This function cleans and extracts words from list
    """
    words = re.sub("[^\w]", " ", sentence).split()
    words_cleaned = [w.lower() for w in words if len(w) > 3]
    return [words_cleaned]


def tokenize_sentence(extracted_data):
    """
    This function tokenize context into array of words
    """
    array_of_words = []
    for data in extracted_data:
        w = word_cleaning_and_extraction(data)
        array_of_words.extend(w)
    return array_of_words


def get_extraction(web_data, page_id):
    """
    This function extract page information
    """
    return [web_data['query']['pages'][str(page_id)]['extract']]


def update_stop_words(stopwords):
    """
    This function use of nltk stop words
    """
    stopwords = stopwords.words('english')
    return stopwords


def get_vectorizer():
    """
    This function return Count Vectorizer constructor
    """
    return CountVectorizer(ngram_range=(1, 1),
                           token_pattern=r'\b[^\d\W]+\b',
                           lowercase=False,
                           stop_words=update_stop_words(stopwords))


def count_frequencies(extracted_data):
    """
    This function count words using Vectorizer
    """
    corpus = tokenize_sentence(extracted_data)
    vectorizer = get_vectorizer()
    return pd.DataFrame(vectorizer.fit_transform(corpus[0]).toarray(), columns=vectorizer.get_feature_names())


if __name__ == "__main__":
    n = 5
    page_id = 21721040

    web_data = get_data_from_url(page_id)

    extracted_data = get_extraction(web_data, page_id)
    word_count = count_frequencies(extracted_data)
    sum_of_appearance = word_count.sum(axis=0).to_frame()
    ordered_of_appearance = sum_of_appearance.sort_values(by=0, ascending=False)
    ordered_of_appearance.rename(columns={0: 'num_of_appearance'}, inplace=True)

    ordered_of_appearance['duplicated'] = ordered_of_appearance['num_of_appearance'].duplicated(keep=False)
    grouped_data = ordered_of_appearance.drop_duplicates('num_of_appearance')
    grouped_data = grouped_data.drop('duplicated', axis=1)
    print("Title: " + str(get_title(web_data, page_id)))
    print("Top " + str(n) + " words:")
    print(grouped_data.head(n))



