# This class provides utilities for processing data through a Dynamic Memory Network

import os as os
import numpy as np
import json
import re
import math

from bs4 import BeautifulSoup
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# init_data using folder name processes the files from the file system.
# This module processes the data into tasks that can be used for training.
# The data is split into information blocks, question blocks, and answer blocks to process
# through the neural network. Returns the data as a list of dictionaries.
def init_data(fname):
    print "==> Loading test from %s" % fname
    tasks = [] # list that will be returned
    documents = ""
    for f in os.listdir(fname):
    	inData = open(fname + "/" + f)
    	for i, line in enumerate(inData):
		line = line.strip()
		try:
			post = json.loads(line) # make sure we can parse the json
		except Exception:
			print(line)
		text = post["body_text"]
		text = text_to_words(text) # call text_to_words to process the text. See text_to_words
		novelty = post["novelty"] 
		task = {"C": "","Q": "", "A": ""} 
		if i < 100:
			documents += text # add the first 100 documents before setting any tasks
		elif i < 200:
			task["C"] += documents # add the next 100 documents as a task with the new document as a question.
			task["Q"] = text
        		task["A"] = novelty
        		tasks.append(task.copy())
			documents += text
    return tasks

# Go fetch and process the raw data from the file system using init_data. See init_data.
def get_raw_data(input_file_train, input_file_test):
    raw_data_train = init_data(input_file_train)
    raw_data_test = init_data(input_file_test)
    return raw_data_train, raw_data_test

# Load glove data for word2vec            
def load_glove(dim):
    word2vec = {}
    
    print "==> loading glove"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print "==> glove is loaded"
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)

def text_to_words(raw_text):
	'''
	Algorithm to convert raw text to a return a clean text string
	Method modified from code available at:
        https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

	Args:
		raw_text: Original text to clean and normalize

	Returns:
		clean_text: Cleaned text
	'''

	# 1. Remove HTML
	#TODO Potentially look into using package other than BeautifulSoup for this step
	#review_text = BeautifulSoup(raw_text, "lxml").get_text()
	#
	# 2. Remove non-letters
	#letters_only = re.sub("[^a-zA-Z]", " ", review_text)
	#
	letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
	# 3. Convert to lower case, split into individual words
	words = letters_only.lower().split()
	#
	# 4. Remove stop words
	meaningful_words = [w for w in words if not w in ENGLISH_STOP_WORDS]
	#
	# 5. Join the words back into one string separated by space,
	# and return the result.
	clean_text = ( " ".join( meaningful_words ))
	return   clean_text
