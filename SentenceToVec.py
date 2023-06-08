from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


class SentenceToVec:
    """
    This class is used to create a vector representation of a sentence (here typically a news headline).
    """
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    def __init__(self, model, only_word2vec = True):
        if only_word2vec:
            self.model = model
        else:
            self.model = model.wv
        self.stop_words = set(stopwords.words('english'))
        self.punct = set(string.punctuation)

    def get_sentence_vector(self, sentence):
        """
        Create a vector representation of a sentence.

        Parameters:
        - sentence: The sentence for which to create the vector representation.

        Returns:
        - A numpy array representing the vector representation of the sentence.
        """
        tokens = word_tokenize(sentence.lower())
        tokens = [token for token in tokens if token not in self.stop_words and token not in self.punct]
        vectors = [self.model[token] for token in tokens if token in self.model]
        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        # Similar results if np.sum is used instead of np.mean
        return np.mean(vectors, axis=0)

    @classmethod
    def fit(cls, sentences):
        """
        Create a Word2Vec model from a list of sentences.

        Parameters:
        - sentences: A list of sentences from which to create the Word2Vec model.

        Returns:
        - An instance of the SentenceToVec class initialized with the created Word2Vec model.
        """
        cleaned_sentences = []
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            tokens = [token for token in tokens if token not in cls.stop_words and token not in cls.punct]
            cleaned_sentences.append(tokens)
        model = Word2Vec(cleaned_sentences, min_count=1, vector_size=20, workers=4, window=5)
        return cls(model.wv, only_word2vec=True)

    @classmethod
    def load(cls, filename, only_word2vec = True):
        """
        Load a Word2Vec model from a file.

        Parameters:
        - filename: The name of the file containing the saved Word2Vec model.

        Returns:
        - An instance of the SentenceToVec class initialized with the loaded Word2Vec model.
        """
        if not only_word2vec:
            model = Word2Vec.load(filename).wv
        else:
            model = KeyedVectors.load(filename)
        return cls(model, only_word2vec=True)

    def save(self, filename):
        """
        Save a Word2Vec model to a file.

        Parameters:
        - filename: The name of the file to which to save the Word2Vec model.
        """
        self.model.save(filename)
