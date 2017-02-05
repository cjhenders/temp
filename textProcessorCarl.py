# -*- coding: utf-8 -*-
'''
Code used to Preprocess data for Kaggle Club Competition- Transfer Learning on Stack Exchange Tags
Ideas come from matteoTosi in Competition Kernel

Carl Henderson 2017
'''

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

dataframes = {
    "cooking": pd.read_csv("/Users/CarlHenderson/Documents/Kaggle Club/Data/Stack Exchange Learning/cooking.csv"),
    "crypto": pd.read_csv("/Users/CarlHenderson/Documents/Kaggle Club/Data/Stack Exchange Learning/crypto.csv"),
    "robotics": pd.read_csv("/Users/CarlHenderson/Documents/Kaggle Club/Data/Stack Exchange Learning/robotics.csv"),
    "biology": pd.read_csv("/Users/CarlHenderson/Documents/Kaggle Club/Data/Stack Exchange Learning/biology.csv"),
    "travel": pd.read_csv("/Users/CarlHenderson/Documents/Kaggle Club/Data/Stack Exchange Learning/travel.csv"),
    "diy": pd.read_csv("/Users/CarlHenderson/Documents/Kaggle Club/Data/Stack Exchange Learning/diy.csv"),
}

print(dataframes["robotics"].iloc[1])


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'


def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""


# This could take a while
for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)


print(dataframes["robotics"].iloc[1])

stops = set(stopwords.words("english"))


def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

for df in dataframes.values():
    df["title"] = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)


for name, df in dataframes.items():
    # Saving to file
    assert isinstance(df.to_csv, object)
    df.to_csv(name + "_light.csv", index=False)