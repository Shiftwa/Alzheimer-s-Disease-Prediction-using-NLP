import gensim
import pandas as pd
# import keras
import pickle
import numpy as np
import tensorflow as tf
import re


from keras.layers import Layer
import keras.backend as K
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional, Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('wordnet')


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(
            input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(
            input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W)+self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x*at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


class attention_lstm_random:
    def __init__(self, optimizer='Adam',
                 dropout_rate=0.2, nb_filters=256, kernel_size=5, pool_size=2, units=128, neurons=20):
        seq_input = tf.keras.Input(shape=(100,))
        vocab_len = 847
        e = Embedding(vocab_len, 100, input_length=100)(seq_input)
        conv1 = Conv1D(filters=256, kernel_size=5, activation='relu',
                       strides=1, kernel_initializer='he_uniform')(e)
        pool = MaxPooling1D(pool_size=4)(conv1)
        dropout = Dropout(0.5)(pool)
        lstm1 = Bidirectional(LSTM(64, return_sequences=True))(dropout)
        att = attention()(lstm1)
        dense1 = Dense(units=128, activation='relu',
                       kernel_initializer='he_uniform')(att)
        output = Dense(units=1, activation='sigmoid',
                       kernel_initializer='glorot_uniform')(dense1)
        att_model = tf.keras.Model(seq_input, output)
        att_model.load_weights('attention_model.hdf5')
        self.att_model = att_model
        self.train_tokenizer = self.create_tokens()

    # function to pad the data to maximum phrase length
    def encode_sentences(self, tokens, length, lines):
        X = tokens.texts_to_sequences(lines)
        X = pad_sequences(X, length, padding='post')
        return X

    def create_tokens(self):
        # Load Tokenizer configuration from JSON file
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            tokens = tokenizer_from_json(tokenizer_json)
        return tokens

    def get_encoded_text(self, text):
        toke = self.train_tokenizer
        encoded_text = self.encode_sentences(toke, 100, text)
        return encoded_text

    def preprocess_text(self, text):
        text = self.preprocess(text)
        preprocessed_data = self.get_encoded_text([text])
        return preprocessed_data

    def preprocess(self, txt):
        preprocessed_data = []
        # remove all the characters except alphabets (E.g. &, *, #)
        df = re.sub('[^a-zA-Z]', ' ', txt)
        df = df.lower()  # convert all the sentences to lower case
        df = df.split()  # tokenize each word
        customStopwords = {'a', 'are', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'any', 'aren', "aren't", 'as',
                           'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
                           'can', 'couldn', "couldn't", 'd', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't",
                           'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't",
                           'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
                           'in', 'into', 'is', 'isn', "isn't", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't",
                           'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of',
                           'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same',
                           'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll",
                           'the', 'there', "there's", 'their', 'theirs', 'them', 'themselves', 'then', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
                           'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when',
                           'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y',
                           'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}
        # removing stopwords
        df = [word for word in df if word not in customStopwords]
        lemmatizer = WordNetLemmatizer()
        df = [lemmatizer.lemmatize(word)
              for word in df]  # performed lemmatization
        # using space as separator to concatenate the list elements
        df = ' '.join(df)
        print(df)

        return df
