# -*- coding: utf-8 -*-
"""Model testing of Sentiment Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rG057eg0qcLMSM9Pme_n760rFCyGeqn5

Problem Statement: This project will perform sentimental analysis on tweets extracted using twitter api. Only 2 labels are included: positive and negative. We will focus on the hyperparameter tuning of bidirectional LSTM layer and model architecture to observe the effects on the resulting model's accuracy. Hyperparameter tunning includes the number of hiddens layers, number of neurons in dense layer, dropout value, loss functions used and etc.

datasource url: https://www.kaggle.com/datasets/kazanova/sentiment140

Sentiment analysis uses text analysis techniques to interpret and categorise emotions (positive, negative, and neutral) in text data. Companies can determine how the general public feels about particular terms or topics by using sentiment analysis.

There are 1.6M tweets in the dataset. In order to reduce the training time, we resize it to 100,000 data.

ref
https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-using-word2vec-bilstm

# Import Library
"""

import pandas as pd
import re
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding

DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('sentiment.csv',encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
dataset.head()

# Removing the unnecessary columns.
dataset = dataset[['sentiment','text']]

# Replacing the values. "0": Negative, "1": Positive
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

# Resample the dataset to 100000
#dataset = resample(dataset, n_samples=100000, random_state=42)

# Check the distribution of data, It is a balanced dataset
#ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',legend=False)
#ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)

dataset.info()

"""#Data Preprocessing"""

# Read contractions.csv and store it as a dictionary.
contractions = pd.read_csv('contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Regex Patterns
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Emoji in Regex
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocessing(tweet):

    tweet = tweet.lower()

    # Replace URls
    tweet = re.sub(urlPattern,'<url>',tweet)
    
    # Replace username to user
    tweet = re.sub(userPattern,'<user>', tweet)
    
    # Replace three or more consecutive letters by two letters
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace emojis
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet

dataset['cleaned_text'] = dataset.text.apply(preprocessing)

"""# Data Spliting"""

from sklearn.model_selection import train_test_split

X_data, y_data = np.array(dataset['cleaned_text']), np.array(dataset['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size = 0.05, random_state = 0)

"""#Create Word Embeddings"""

from gensim.models import Word2Vec

Embedding_dimensions = 100

# Creating Word2Vec training dataset
Word2vec_train_data = list(map(lambda x: x.split(), X_train))

# Define word2vec model
word2vec_model = Word2Vec(Word2vec_train_data,
                 workers= 8,
                 min_count= 5)

"""#Tokenization"""

# Defining model input length
input_length = 60

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_length = 60000

tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = vocab_length

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape :", X_test.shape)

"""#Create Embedding Matrix"""

embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

print("Embedding Matrix Shape:", embedding_matrix.shape)

"""# Model 1

###LSTM: 100 neurons, Dropout = 0.3, Batch size: 128, Epoch: 10
"""

def Model():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

training_model = Model()
training_model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

fit_model = training_model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

score_1 = training_model.evaluate(X_train, y_train, verbose=1)
score_1

print("Train Loss:", score_1[0])
print("Train Accuracy:", score_1[1])

plt.plot(fit_model.history['accuracy'])
plt.plot(fit_model.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(fit_model.history['loss'])
plt.plot(fit_model.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def ConfusionMatrix(y_pred, y_test):
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['TN','FP', 'FN','TP']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

# Predicting on the Test dataset
y_pred = training_model.predict(X_test)

# Converting prediction to reflect the sentiment predicted
y_pred = np.where(y_pred>=0.5, 1, 0)

# Evaluation metrics
ConfusionMatrix(y_pred, y_test)

# Evaluation metrics
print(classification_report(y_test, y_pred))

"""# Model 3

LSTM (add one more dense layer): 100 neurons, Dropout = 0.3, Batch size: 128, Epoch: 10
"""

def Model3():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

training_model = Model3()
training_model.summary()

training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_model = training_model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

score_2 = training_model.evaluate(X_train, y_train, verbose=1)
print("Train Loss:", score_2[0])
print("Train Accuracy:", score_2[1])

plt.plot(fit_model.history['accuracy'])
plt.plot(fit_model.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(fit_model.history['loss'])
plt.plot(fit_model.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

"""# Summary"""

