# -*- coding: UTF-8 -*-

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from bs4 import BeautifulSoup
import re
import string
import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'


def process_text(text):
	stop_words = set(stopwords.words('english'))
	stemmer = SnowballStemmer("english")
	#toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
	text = re.sub(r'[^\x00-\x7f]',r' ',text)
	text = re.sub("["+string.punctuation+"]", " ", text)
	#word_tokens = tokenizer.tokenize(text)
	#word_tokens = [x for x in word_tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
	filtered_sentence = [stemmer.stem(w) for w in text.split() if w not in stop_words]
	#filtered_sentence = [ for w in filtered_sentence]


	return " ".join(filtered_sentence)

#example_sent = "This is a sample sentence, showing off the stop words filtration."
#ex2 = '''He said,"that's it." *u* Hello, World.'''
#print process_text(ex2)


def remove_html(values):
        soup = BeautifulSoup(values, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)

def run_on_data():
	data = {
	    "cooking": pd.read_csv("../kaggle/inputs/cooking.csv"),
	    "biology": pd.read_csv("../kaggle/inputs/biology.csv")
	}

	for df in data.values():
	    df["content"] = df["content"].map(remove_html)
	    df['title'] = df['title'].map(process_text)
	    df["content"] = df["content"].map(process_text)
	    df["tags"] = df["tags"].map(lambda x: x.split())

	print(data["cooking"].iloc[0])
	for name, df in data.items():
		df.to_csv(name + "_light.csv", index=False)
    	# Saving to file


def tf(word, blob):
    return (float)(blob.words.count(word)) / (float)(len(blob.words))

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (float)(1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

# bloblist = [document1, document2, document3]
# for i, blob in enumerate(bloblist):
#     print("Top words in document {}".format(i + 1))
#     scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
#     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     for word, score in sorted_words[:3]:
#         print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))

data = {
	    "cooking": pd.read_csv("../kaggle/cooking_light.csv"),
	    "biology": pd.read_csv("../kaggle/biology_light.csv")
	}

train_set = data.values() #Documents
test_set = ["The sun in the sky is bright."] #Query
stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()

trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()
print 'Fit Vectorizer to train set', trainVectorizerArray
print 'Transform Vectorizer to test set', testVectorizerArray

transformer.fit(trainVectorizerArray)
print transformer.transform(trainVectorizerArray).toarray()

transformer.fit(testVectorizerArray)

tfidf = transformer.transform(testVectorizerArray)
print tfidf.todense()