# Sentence-level-Classification-using-Naive-Bayes
I have used a real-time dataset from the Machine Learning Repository of Center for Machine Learning and Intelligent Systems at UCI to perfrom sentence-level classification. The corpus contains 972 annotated sentences from the abstract and introduction of 30 scientific articles for the training purpose. Naive Bayes was used as the classifier for the task. The classifer then classified 700+ sentences from over 300 scientific articles with an accuracy of over 60 percent. I used python 2.7 notebook for my implementation.

###importing the required libraries

import numpy as np
import collections
import sys
import csv
import os
from os import listdir
import re
import nltk.data
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.tokenize import LineTokenizer
from nltk.corpus import stopwords 
from __future__ import division

###defining a function to extract and tokenize individual sentences from the corpus

def tokenize_sentences(Directory):
    label = {}
    sentence = {}
    normalized_sentence = {}
    features = {}
    global train
    k = 0
    for filename in listdir(Directory):          #Reading each file from the corpus
        if filename.endswith(".txt"):
            f = open(filename,'rb')
            data = f.read()
            i = 0;
            while data is not None:
                train = LineTokenizer(blanklines = 'discard').tokenize(data)  #tokenizing the text files
                try:
                    # preparing the training dataset with the extracted sentences and the corresponding labels
                    if any(c in train[i][:4] for c in ('MISC','OWNX','BASE','CONT','AIMX')):
                        label[k] = train[i][:4]
                        sentence[k] = train[i][5:][0:]
                        # removing the stopwords from the sentences using the stopwords list of nltk library
                        normalized_sentence[k] = ' '.join([w for w in re.split('\W',sentence[k]) if w.lower() not in stopwords.words('english') and w.lower() != ''])
                except IndexError:
                    break 
                i += 1
                k += 1
     return label, normalized_sentence
    
###Function to store the tokenized training data in a CSV file

def export_training_data_to_csv(filename):
    training_set = {}
    with open(filename,'wb') as csvfile:
            fieldnames = ['normalized_sentence','label']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            for i in range(1,len(label),1):
                try:
                    writer.writerow({'normalized_sentence': normalized_sentence[i],'label': label[i]})
                except IndexError:
                    continue   
                except KeyError:
                    continue
            sys.stdout.flush()
            os.fsync(csvfile.fileno())
            csvfile.close()
            
###Setting the Directory for the corpus
Directory = "E:/binesh/MSc in Computing/Semester 2/Machine Learning/Datasets/SentenceCorpus/labeled_articles/training_dataset/"

###Setting the tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
os.chdir(r'E:/binesh/MSc in Computing/Semester 2/Machine Learning/Datasets/SentenceCorpus/labeled_articles/training_dataset')

###A call to the pre-defined function to tokenize the sentences
label, normalized_sentence = tokenize_sentences(Directory)

###A function call to store the tokenized data in a CSV file named, training_data.csv
export_training_data_to_csv('training_data.csv')

###Training the Naive Bayes Classifier

with open('training_data.csv','rb') as csvfile:
    C = NaiveBayesClassifier(csvfile,format = "csv")
csvfile.close()

###Setting the directory for the test dataset
Directory = "E:/binesh/MSc in Computing/Semester 2/Machine Learning/Datasets/SentenceCorpus/labeled_articles/test_dataset/"
os.chdir(r'E:/binesh/MSc in Computing/Semester 2/Machine Learning/Datasets/SentenceCorpus/labeled_articles/test_dataset/')

###Function call to tokenize the test dataset 
label, normalized_sentence = tokenize_sentences(Directory)
###Storing the tokenized test dataset in a CSV file test_data.csv
export_training_data_to_csv('test_data.csv')

###Computing the accuracy of the Classifier using 30% of the labeled corpus as the test data

accuracy = 0
line_number = 0
with open('test_data.csv','rb') as csvfile:
        reader = csv.reader(csvfile)
        reader = list(reader)
        for filename in listdir(Directory):
            if (filename.endswith(".txt")):
                f = open(filename, 'rb')
                data = f.readlines()
                for sentence in data:
                    # removing the stopwords from the sentences using the stopwords list of nltk library
                    normalized_sentence = ' '.join([w for w in re.split('\W',sentence) if w.lower() not in stopwords.words('english')])
                    # Classifying the individual sentences
                    blob = TextBlob(normalized_sentence, classifier = C)
                    if line_number < len(reader):
                    # Comparing the classified label with the pre-defined label in the test dataset
                        if reader[line_number][1] == blob.classify():
                        # Incrementing the accuracy for each match
                            accuracy = accuracy + 1                    
                        line_number += 1
        csvfile.close()
print format((accuracy/len(reader))* 100, '.2f') # Printing the accuracy of the Classifier

###Classifying unknown sentences and writing the results to a CSV file

Directory = 'E:/binesh/MSc in Computing/Semester 2/Machine Learning/Datasets/SentenceCorpus/unlabeled_articles/unknown_sample/'
os.chdir(r'E:/binesh/MSc in Computing/Semester 2/Machine Learning/Datasets/SentenceCorpus/unlabeled_articles/unknown_sample/')
with open('classified_sentences.csv','wb') as csvfile:
    fieldnames = ['Sentence', 'Label']
    writer = csv.DictWriter(csvfile,fieldnames = fieldnames)
    writer.writeheader()
    for filename in listdir(Directory):
        if filename.endswith(".txt"):
            f = open(filename, 'rb')
            data = f.readlines()
            for sentence in data:
                if not(sentence.startswith('###')):
                    # removing the stopwords from the sentences using the stopwords list of nltk library
                    normalized_sentence = ' '.join([w for w in re.split('\W',sentence) if w.lower() not in stopwords.words('english')])
                    # Classifying the individual sentences
                    blob = TextBlob(normalized_sentence, classifier = C)
                    # Writing the sentences along with their classified label to the CSV file
                    writer.writerow({'Sentence': sentence, 'Label': blob.classify()})
csvfile.close()
