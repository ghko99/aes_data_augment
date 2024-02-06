import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import csv
# kobert
# tensorflow & keras & sklearn
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras.utils
import math
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import cohen_kappa_score
import argparse

def preprocessing(essay_data_origin):
    essay_data = []
    for data in essay_data_origin:
        new = data.replace('<span>', '').replace(
            '</span>', '').replace('\n', '').replace('\t', '')
        essay_data.append(new)
    return essay_data

def essay_to_sentences(essay_v):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    raw_sentences = essay_v.split('#@문장구분#')
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(raw_sentence)
    return sentences

def get_y(X):
    y = []
    for i in tqdm(range(len(X))):
         y.append([round(X['exp1'][i]/3, 2), round(X['exp2'][i]/3, 2), round(X['exp3'][i]/3, 2), round(X['org1'][i]/3, 2),
                           round(X['org2'][i]/3, 2), round(X['org3'][i]/3, 2), round(X['org4'][i]/3, 2), round(X['con1'][i]/3, 2), round(X['con2'][i]/3, 2), round(X['con3'][i]/3, 2), round(X['con4'][i]/3, 2)])
    y = pd.DataFrame(y)
    return y

def get_essays_y():
    DATASET_DIR = './data/'
    SAVE_DIR = './'
    X = pd.read_csv(os.path.join(
        DATASET_DIR, 'dataset.csv'), encoding='utf-8')
    
    essay_data_origin = X['ESSAY_CONTENT']
    preprocessed_essays = preprocessing(essay_data_origin)
    essays = []
    for ix, essay in enumerate(preprocessed_essays):
        sentences = essay_to_sentences(essay)
        essays.append(sentences)
    y = get_y(X)

    return essays, y

def get_essays_y_with_topic():
    DATASET_DIR = './data/'
    SAVE_DIR = './'
    X = pd.read_csv(os.path.join(
        DATASET_DIR, 'dataset.csv'), encoding='utf-8')
    
    essay_data_origin = X['ESSAY_CONTENT']
    essay_data_topic = X['ESSAY_SUBJECT']
    preprocessed_essays = preprocessing(essay_data_origin)
    essays = []
    for ix, essay in enumerate(preprocessed_essays):
        sentences = essay_to_sentences(essay)
        topic = essay_data_topic.iloc[ix]
        sentences_with_topic =  [item for sentence in sentences for item in [topic, sentence]]
        essays.append(sentences_with_topic)
    y = get_y(X)
    return essays, y

def get_embedded_essay(essays):
    # read_embedded_data
    DATASET_DIR = './data/'
    embedded_essay_raw = pd.read_csv(os.path.join(
        DATASET_DIR, 'embedded_features_kobert_holistic.csv'), encoding='cp949')
    embedded_essay = []
    print(embedded_essay_raw.shape)
    tmp_ix = 0
    for ix, essay_raw in enumerate(essays):
        tmp_len = len(essay_raw)
        essay = embedded_essay_raw[tmp_ix:tmp_ix + tmp_len]
        embedded_essay.append(essay)
        tmp_ix += tmp_len
    
    return embedded_essay


def get_sentence_model(n_outputs):
    """Define the model."""
    model = Sequential()
    model.add(GRU(256, dropout=0.4, input_shape=[
              128, 768], return_sequences=True))
    model.add(GRU(128, dropout=0.4, return_sequences=True))
    model.add(GRU(64))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.summary()

    return model

def get_y(X):
    y = []
    for i in tqdm(range(len(X))):
         y.append([round(X['exp1'][i]/3, 2), round(X['exp2'][i]/3, 2), round(X['exp3'][i]/3, 2), round(X['org1'][i]/3, 2),
                           round(X['org2'][i]/3, 2), round(X['org3'][i]/3, 2), round(X['org4'][i]/3, 2), round(X['cont1'][i]/3, 2), round(X['cont2'][i]/3, 2), round(X['cont3'][i]/3, 2), round(X['cont4'][i]/3, 2)])
    y = pd.DataFrame(y)
    return y

# data generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self,embedded_essay,y, ids, batch_size=64, shuffle=True):
        self.embedded_essay = embedded_essay
        self.y =y
        self.ids = ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

    def __getitem__(self, index):
        # Generated data containing batch_size samples
        batch_ids = self.ids[index *
                             self.batch_size:(index + 1) * self.batch_size]

        essays = list()
        scores = list()
        for ix in batch_ids:
            essay = self.embedded_essay[ix]
            score = self.y.iloc[ix]
            essays.append(essay)
            scores.append(score)
        essays = pad_sequences(
            essays, maxlen=128, padding='pre', dtype='float')

        return np.array(essays), np.array(scores)

def get_dataset(embedded_essay, y,DATASET_DIR,batch_size):

    train_ids = pd.read_csv(os.path.join(
        DATASET_DIR, 'trainset.csv'), encoding='cp949')
    train_ids_list = train_ids['ID'].tolist()

    test_ids = pd.read_csv(os.path.join(
        DATASET_DIR, 'testset.csv'), encoding='cp949')
    test_ids_list = test_ids['ID'].tolist()


    train_gen = DataGenerator(embedded_essay,y,train_ids_list, batch_size=batch_size)
    test_gen = DataGenerator(embedded_essay,y,test_ids_list,batch_size=batch_size,shuffle=False)

    return train_gen, test_gen

def aes_train_test(n_outputs,batch_size, is_topic):
    if is_topic == True:
        essays, y = get_essays_y_with_topic()
    else:
        essays, y = get_essays_y()

    embedded_essay = get_embedded_essay(essays)
    train_gen,test_gen = get_dataset(embedded_essay,y,'./data/',batch_size)

    train_steps = train_gen.__len__()
    
    early_stopping = EarlyStopping(
        monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
    sentence_model = get_sentence_model(n_outputs)
    sentence_model.fit(train_gen, steps_per_epoch=train_steps,epochs=50, callbacks=[early_stopping])
    sentence_model.save('./data/kobert_model.h5')
    
    y_sent_pred = sentence_model.predict(test_gen) * 100
    y_sent_pred = np.round(y_sent_pred)
    y_sent_pred = np.array(y_sent_pred)

    test_ids_list = pd.read_csv('./data/testset.csv',encoding='cp949')['ID'].to_list()
    test_y = y.iloc[test_ids_list].values
    y_test = np.array(np.round(test_y*100))

    test_x = np.array(pd.read_csv('./data./dataset.csv',encoding='utf-8-sig').iloc[test_ids_list].reset_index(drop=True))

    return y_sent_pred, y_test, test_x


def compute_metrics(y_sent_pred,y_test, test_x, cnt):
    
    rubric = pd.read_csv(os.path.join('./data/rubric.csv'), encoding='utf-8-sig')
    weighted_pred_list, weighted_real_list = [], []

    result_file = os.path.join('./data/bert_result.csv')
    

    # get total score per essay
    for i in range(len(y_test)):
        tmp_rubric = rubric.loc[(
            rubric['SUBJECT'] == test_x[i][2])].to_numpy()[0]
        tmp_exp = (y_sent_pred[i][0] * tmp_rubric[6] + y_sent_pred[i][1] * tmp_rubric[7] + y_sent_pred[i][2] * tmp_rubric[8]) \
            / (tmp_rubric[6] + tmp_rubric[7] + tmp_rubric[8])
        tmp_org = (y_sent_pred[i][3] * tmp_rubric[10] + y_sent_pred[i][4] * tmp_rubric[11] + y_sent_pred[i][5] * tmp_rubric[12] + y_sent_pred[i][6] * tmp_rubric[13]) \
            / (tmp_rubric[10] + tmp_rubric[11] + tmp_rubric[12] + tmp_rubric[13])
        tmp_con = (y_sent_pred[i][7] * tmp_rubric[15] + y_sent_pred[i][8] * tmp_rubric[16] + y_sent_pred[i][9] * tmp_rubric[17] + y_sent_pred[i][10] * tmp_rubric[18]) \
            / (tmp_rubric[15] + tmp_rubric[16] + tmp_rubric[17] + tmp_rubric[18])
        tmp_pred_score = (
            tmp_exp * tmp_rubric[5] + tmp_org * tmp_rubric[9] + tmp_con * tmp_rubric[14]) / 10
        weighted_pred_list.append(tmp_pred_score)

        tmp_exp = (y_test[i][0] * tmp_rubric[6] + y_test[i][1] * tmp_rubric[7] + y_test[i][2] * tmp_rubric[8]) \
            / (tmp_rubric[6] + tmp_rubric[7] + tmp_rubric[8])
        tmp_org = (y_test[i][3] * tmp_rubric[10] + y_test[i][4] * tmp_rubric[11] + y_test[i][5] * tmp_rubric[12] + y_test[i][6] * tmp_rubric[13]) \
            / (tmp_rubric[10] + tmp_rubric[11] + tmp_rubric[12] + tmp_rubric[13])
        tmp_con = (y_test[i][7] * tmp_rubric[15] + y_test[i][8] * tmp_rubric[16] + y_test[i][9] * tmp_rubric[17] + y_test[i][10] * tmp_rubric[18]) \
            / (tmp_rubric[15] + tmp_rubric[16] + tmp_rubric[17] + tmp_rubric[18])
        tmp_real_score = (
            tmp_exp * tmp_rubric[5] + tmp_org * tmp_rubric[9] + tmp_con * tmp_rubric[14]) / 10
        weighted_real_list.append(tmp_real_score)
        
    ff = open(result_file, 'w', newline='')
    writer_ff = csv.writer(ff)
    writer_ff.writerow(['ESSAY_ID', 'REAL_SCORE', 'PRED_SCORE'])
    for i in range(len(weighted_pred_list)):
        writer_ff.writerow([test_x[i][0], weighted_real_list[i], weighted_pred_list[i]])
    ff.close()
    # print kappa, pearson score
    sentence_result = cohen_kappa_score(
    np.round(weighted_real_list), np.round(weighted_pred_list), weights='quadratic')
    pearson_result = np.corrcoef(
        np.round(weighted_real_list), np.round(weighted_pred_list))[0, 1]
    print("Kappa Score", cnt, ": {}".format(sentence_result))
    print("Pearson Correlation Coefficient",
        cnt, ": {}".format(pearson_result))

def main():
    
    parser = argparse.ArgumentParser(description='insert topic argument parser')
    parser.add_argument('--is_topic',  
                        type=lambda s : s.lower() in ['true','1'], 
                        help='Boolean argument about inserting topic')
    
    args = parser.parse_args()

    is_topic = args.is_topic

    cnt = 1
    n_outputs = 11
    batch_size =  64
    
    y_sent_pred, y_test,test_x = aes_train_test(n_outputs=n_outputs, batch_size=batch_size,is_topic=is_topic)
    compute_metrics(y_sent_pred,y_test,test_x,cnt)

if __name__ == '__main__':
    main()
