# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:21:10 2019

@author: tdpco
"""

# Importing Libraries
import spacy
import numpy as np
import random
from pickle import dump,load
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

# Function Definitions


def read_file(filepath):
    '''
    Function to read file from a given path
    '''
    with open(filepath) as f:
        str_text = f.read()
    return str_text


def seperate_punc(doc_text):
    '''
    Function to take only document words that are not punctuation
    '''
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


def create_model(vocabulary_size, seq_len):
    '''
    Function to create a LSTM model
    '''
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    '''
    Function to predict next word
    '''
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' '+pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)

# Loading english language
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
nlp.max_length = 1198623

# Loading the dataset
dataset = read_file('moby_dick_four_chapters.txt')
print("\nThe text from our Dataset : \n")
print(dataset[0:500])
print("\n")
tokens = seperate_punc(dataset)
training_length = 25+1
text_sequences = []

# Creating a batch of sequences with 26 words
for i in range(training_length, len(tokens)):
    seq = tokens[i-training_length:i]
    text_sequences.append(seq)
print("Dividing the chapters into sequences of 26 words - \n\n")
print(f"Text Sequence 1 - \n {' '.join(text_sequences[0])} \n")
print(f"Text Sequence 2 - \n {' '.join(text_sequences[1])} \n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
vocabulary_size = len(tokenizer.word_counts)
sequences = np.array(sequences)

# Divinding into X and y data
X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocabulary_size+1)
seq_len = X.shape[1]

# Creating a model by calling create_model method
model = create_model(vocabulary_size+1, seq_len)

# Fitting the model
model.fit(X, y, batch_size=128, epochs=8, verbose=1)

# Saving the model and tokenizer
model.save('my_mobydick_model.h5')
dump(tokenizer, open('my_simpletokenizer', 'wb'))

# creating a random sequence
random.seed(101)
random_pick = random.randint(0, len(text_sequences))
random_seed_text = text_sequences[random_pick]
seed_text = ' '.join(random_seed_text)
pred_words = generate_text(model, tokenizer, seq_len, seed_text, num_gen_words=15)
print(f"\n\n The input text : \n {seed_text}")
print(f"\n\n The predicted text (next 15 words) is : \n {pred_words}")
seed_text = ' '.join(text_sequences[2565])
pred_words = generate_text(model, tokenizer, seq_len, seed_text, num_gen_words=25)
print(f"\n\n The input text : \n {seed_text}")
print(f"\n\n The predicted text (next 25 words) is : \n {pred_words}")
