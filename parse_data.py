import xml.etree.ElementTree as ET
import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict


def clean_up(text):
    #convert to lower-case
    text = text.lower()
    #remove numbers
    text = re.sub(r'\d+', '', text)
    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

def parse_review(filepath, in_vocab=None):
    tree = ET.parse(filepath)
    root = tree.getroot()

    vocab = defaultdict(int)
    tokenized_reviews = []
    labels = []
    clean_reviews = []

    for review in root.findall('review'):
        #grab review text and star rating for each review
        text = review.find('review_text').text
        stars = int(float(review.find('rating').text.strip('\n')))

        #convert star rating to pos/neg label and add to list of labels
        if stars == 3: continue
        label = 1 if stars > 3 else -1
        labels.append(label)

        #clean up review text 
        clean_text = clean_up(text)
        clean_reviews.append(clean_text)

        #tokenize review text
        tokens = word_tokenize(clean_text)

        #remove stop words
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

        #add every token to the vocab counts
        for token in tokens:
            vocab[token] += 1

        #add the tokenized review to the list
        tokenized_reviews += tokens

    #removes words that appear less than 3 times from vocabulary 
    for word_type in vocab:
        if vocab[word_type] < 3: 
            vocab[word_type] = 0
    #subsequently removes such words from the reviews
    tokenized_without_outliers = []
    for review in tokenized_reviews:
        review = [token for token in review if vocab[token] > 0]
        tokenized_without_outliers.append(review)

    reviews = tokenized_without_outliers
    

    if in_vocab:
        cv = CountVectorizer(binary=True, vocabulary=in_vocab)
    else:
        cv = CountVectorizer(binary=True)
    cv.fit(clean_reviews)
    vocab = cv.vocabulary_
    X = cv.transform(clean_reviews)
    Y = labels
    return X, Y, vocab



