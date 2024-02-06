import os
import torch
import pandas as pd
import csv
from tensorflow.keras.preprocessing.sequence import pad_sequences
# kobert
from transformers import AutoModel
from tokenization_kobert import KoBertTokenizer
from tqdm import tqdm
import argparse


def load_model_tokenizer():
    tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert").cuda()
    
    return model, tokenizer

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

def get_essays():
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
    
    return essays

def get_essays_with_topic():
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
    return essays

def embedding(tokenizer, model, essays):
    DATASET_DIR = './data/'
    ff = open(os.path.join(
        DATASET_DIR, 'embedded_features_kobert_holistic.csv'), 'w', newline='')
    writer_ff = csv.writer(ff)
    sent_max_len = 50
    for ix in tqdm(range(len(essays))):
        inputs = tokenizer.batch_encode_plus(essays[ix])
        ids_new = pad_sequences(inputs['input_ids'],
                                maxlen=sent_max_len, padding='post')
        mask_new = pad_sequences(
            inputs['attention_mask'], maxlen=sent_max_len, padding='post')
        out = model(input_ids=torch.tensor(ids_new).cuda(),
                    attention_mask=torch.tensor(mask_new).cuda())
        embedded_features = out[0].detach().cpu()[:, 0, :].numpy()
        for i in embedded_features:
            writer_ff.writerow(i)
        torch.cuda.empty_cache()
    ff.close()

def main():
    parser = argparse.ArgumentParser(description='insert topic argument parser')
    parser.add_argument('--is_topic',  
                        type=lambda s : s.lower() in ['true','1'], 
                        help='Boolean argument about inserting topic')
    
    args = parser.parse_args()

    is_topic = args.is_topic
    #load model and tokenizer
    print(is_topic)
    model,tokenizer = load_model_tokenizer()
    #get essays
    if is_topic == True:
        essays = get_essays_with_topic()
    else:
        essays = get_essays()
    print(essays[:5])
    embedding(tokenizer,model,essays)

if __name__ == '__main__':
    main()