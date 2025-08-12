#!/usr/bin/python

import re
from string import punctuation
import nltk
nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

class TextPreProcess():
   def __init__(self):
      re_exp_punctuation = '[{}]'.format('\\'.join([char_ for char_ in punctuation]))
      self.reObjPunct = re.compile(re_exp_punctuation)
      self.reObjWhiteSpace = re.compile(r'\s{2, 10}')

      self.wnl = WordNetLemmatizer()

   def txt_vectorization(self, sequence):
      sequence = sequence.lower()
      sequence = self.reObjPunct.sub(' ', sequence)
      sequence = self.reObjWhiteSpace.sub(' ', sequence)
      wordslist = nltk.word_tokenize(sequence)
      wordslist = [self.wnl.lemmatize(word) for word in wordslist if word not in stopwords.words('english')]
      return wordslist
