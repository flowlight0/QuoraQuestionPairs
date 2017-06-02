"""
Example of an LSTM model with GloVe embeddings along with magic features

Tested under Keras 2.0 with Tensorflow 1.0 backend

Single model may achieve LB scores at around 0.18+, average ensembles can get 0.17+
"""

########################################
# import packages
########################################
import codecs
import csv
import os
import re
import sys
from collections import defaultdict

import gensim
import joblib
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

########################################
# set directories and parameters
########################################
BASE_DIR = os.path.join(os.path.dirname(__file__), '../data/input/')
EMBEDDING_FILE = BASE_DIR + 'glove.840B.300d.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

TRAIN_DATA_CACHE_FILE_1 = BASE_DIR + 'train.csv.1.cache.pkl'
TRAIN_DATA_CACHE_FILE_2 = BASE_DIR + 'train.csv.2.cache.pkl'
TEST_DATA_CACHE_FILE_1 = BASE_DIR + 'test.csv.1.cache.pkl'
TEST_DATA_CACHE_FILE_2 = BASE_DIR + 'test.csv.2.cache.pkl'
TOKENIZE_CACHE_FILE = BASE_DIR + 'tokenize.cache.pkl'


MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
N_SPLITS = 5

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_without_leak_%d_%d_%.2f_%.2f_%d' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, MAX_SEQUENCE_LENGTH)
print(STAMP)


def calculate_glove_texts():
    if os.path.exists(TRAIN_DATA_CACHE_FILE_1) and os.path.exists(TRAIN_DATA_CACHE_FILE_2) and \
            os.path.exists(TEST_DATA_CACHE_FILE_1) and os.path.exists(TEST_DATA_CACHE_FILE_2):
        print("GLove texts are already cached :)", file=sys.stderr)
        return joblib.load(TRAIN_DATA_CACHE_FILE_1), joblib.load(TRAIN_DATA_CACHE_FILE_2), \
               joblib.load(TEST_DATA_CACHE_FILE_1), joblib.load(TEST_DATA_CACHE_FILE_2)

    ########################################
    # process texts in datasets
    ########################################
    print('Processing text dataset')

    # The function "text_to_wordlist" is from
    # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
    def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        text = text.lower().split()

        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]

        text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return text

    texts_1 = []
    texts_2 = []
    labels = []
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts_1.append(text_to_wordlist(values[3]))
            texts_2.append(text_to_wordlist(values[4]))
            labels.append(int(values[5]))
    print('Found %s texts in train.csv' % len(texts_1))

    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_texts_1.append(text_to_wordlist(values[1]))
            test_texts_2.append(text_to_wordlist(values[2]))
            test_ids.append(values[0])

    print('Found %s texts in test.csv' % len(test_texts_1))
    joblib.dump(texts_1, TRAIN_DATA_CACHE_FILE_1)
    joblib.dump(texts_2, TRAIN_DATA_CACHE_FILE_2)
    joblib.dump(test_texts_1, TEST_DATA_CACHE_FILE_1)
    joblib.dump(test_texts_2, TEST_DATA_CACHE_FILE_2)
    return texts_1, texts_2, test_texts_1, test_texts_2


def tokenize():
    if os.path.exists(TOKENIZE_CACHE_FILE):
        print("Tokenization results are already cached :)", file=sys.stderr)
        return joblib.load(TOKENIZE_CACHE_FILE)
    texts_1, texts_2, test_texts_1, test_texts_2 = calculate_glove_texts()
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
    joblib.dump((sequences_1, sequences_2, test_sequences_1, test_sequences_2, tokenizer), TOKENIZE_CACHE_FILE)
    return sequences_1, sequences_2, test_sequences_1, test_sequences_2, tokenizer


sequences_1, sequences_2, test_sequences_1, test_sequences_2, tokenizer = tokenize()

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(pd.read_csv(TRAIN_DATA_FILE).is_duplicate.astype(int).tolist())
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(pd.read_csv(TEST_DATA_FILE).test_id.astype(int).tolist())


train_df, test_df = pd.read_csv(TRAIN_DATA_FILE), pd.read_csv(TEST_DATA_FILE)

########################################
# prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
    except:
        continue
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=114514)
bst_val_scores = []
test_preds = np.zeros(test_df.shape[0])
train_df['prediction'] = np.zeros(test_df.shape[0])

for idx_train, idx_val in skf.split(X=train_df, y=labels):
    ########################################
    # sample train/validation data
    ########################################
    data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
    data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
    labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

    data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
    data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
    labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val == 0] = 1.309028344

    ########################################
    # define the model structure
    ########################################
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## add class weight
    ########################################
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    ########################################
    # train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    # model.summary()
    print(STAMP)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([data_1_train, data_2_train], labels_train,
                     validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                     epochs=200, batch_size=2048, shuffle=True,
                     class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    bst_val_scores.append(bst_val_score)

    ########################################
    # make the submission
    ########################################
    print('Start making prediction on test dataset')
    internal_test_preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
    internal_test_preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
    internal_test_preds /= 2
    test_preds += internal_test_preds.ravel() / N_SPLITS

    val_preds = model.predict([data_1[idx_val], data_2[idx_val]], batch_size=8192, verbose=1)
    val_preds += model.predict([data_2[idx_val], data_1[idx_val]], batch_size=8192, verbose=1)
    val_preds /= 2
    train_df.ix[idx_val, 'prediction'] = val_preds.ravel()

train_output_file = os.path.join(os.path.dirname(__file__), '../data/lstm/',
                                 '%.4f_' % (np.mean(bst_val_scores)) + STAMP + '.stacking.csv')
test_output_file = os.path.join(os.path.dirname(__file__), '../data/lstm/',
                                '%.4f_' % (np.mean(bst_val_scores)) + STAMP + '.submission.csv')
submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': test_preds.ravel()})
submission.to_csv(test_output_file, index=False)
train_df[['id', 'prediction']].to_csv(train_output_file, index=False)
