import numpy as np
import pandas as pd
import os
import pickle
from nltk.translate.bleu_score import sentence_bleu
import json
from keras import backend as K
from numpy import *
import keras
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
from nltk.tokenize import RegexpTokenizer
#import spacy
#from spacy.cli.download import download
#download(model="en_core_web_sm")

max_answer_len = 2048
max_question_len = 30

import sklearn
file_path = "data/version_3.json"

with open(file_path, 'r') as j:
    data = json.loads(j.read())

target_column = []
source_column = []
for key in data['mmcs'].keys():
    target_column.append(data['mmcs'][key])
    source_column.append(data['concated_transcript'][key])

trn_x, val_x, trn_y, val_y = sklearn.model_selection.train_test_split(source_column, target_column,
                                                                test_size=0.2,
                                                                random_state=42)


trn_df = pd.DataFrame(list(zip(trn_x, trn_y)),
               columns =['input', 'output'])

val_df = pd.DataFrame(list(zip(val_x, val_y)),
               columns =['input', 'output'])

def text_cleaning(text):

    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub("http[s]?\://\S+","",text) # removing urls
    text = re.sub("(\s+)",' ', text)
    text = re.sub(r'[^\w\s]', '', text) # removing all punctuation
    text = text.lower()

    return text

train_question_list=[]
train_answer_list=[]
max=0
sum=0
for i in range(len(trn_df)):
    question = text_cleaning(trn_df['output'][i])

    question = 'sostok ' + 'start ' + question + ' end ' + 'eostok'
    train_question_list.append(question)

    answer = trn_df['input'][i]
    train_answer_list.append(answer)

val_question_list=[]
val_answer_list=[]
max=0
sum=0
for i in range(len(val_df)):
    question = text_cleaning(val_df['output'][i])
    question = 'sostok ' + 'start ' + question + ' end ' + 'eostok'
    val_question_list.append(question)

    answer = val_df['input'][i]
    val_answer_list.append(answer)

from tqdm import tqdm

#Importing GloVe embedding
path_to_glove_file = os.path.join(
    os.path.expanduser("~"), "monku/version_3/glove_embeddings/glove.6B.300d.txt"
)

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in tqdm(f):
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

total_string = '  '.join(list(set(train_question_list)))

tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
words = tokenizer.tokenize(total_string)
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

num_tokens = len(unique_words) + 2
embedding_dim = 300
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in unique_word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and embedding_vector.shape!=(0,):
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

answer=train_answer_list
question=train_question_list

train_answer_arr = np.zeros((len(answer), max_answer_len), dtype=int)

for i in range(len(answer)):

  if len(answer[i].split()) < max_answer_len:
    length = len(answer[i].split())

  else:
    length = max_answer_len

    for j in range(len(train_answer_arr)):
        tmp_list=answer[i].split()

        if tmp_list[j] in unique_word_index.keys():
            train_answer_arr[i][j] = unique_word_index[tmp_list[j]]
        else:
            train_answer_arr[i][j] = 0

train_question_arr = np.zeros((len(question), max_question_len), dtype=int)
for i in range(len(question)):

  if len(question[i].split()) < max_question_len:
    length = len(question[i].split())

  else:
    length = max_question_len

  for j in range(length):
    tmp_list=question[i].split()

    if tmp_list[j] in unique_word_index.keys():
      train_question_arr[i][j] = unique_word_index[tmp_list[j]]
    else:
      train_question_arr[i][j] = 0

answer=val_answer_list
question=val_question_list

val_answer_arr = np.zeros((len(answer), max_answer_len), dtype=int)
for i in range(len(answer)):

  if len(answer[i].split()) < max_answer_len:
    length = len(answer[i].split())

  else:
    length = max_answer_len

  for j in range(length):
    tmp_list=answer[i].split()
    if tmp_list[j] in unique_word_index.keys():
      val_answer_arr[i][j] = unique_word_index[tmp_list[j]]
    else:
      val_answer_arr[i][j] = 0

val_question_arr = np.zeros((len(question), max_question_len), dtype=int)
for i in range(len(question)):

  if len(question[i].split()) < max_question_len:
    length = len(question[i].split())

  else:
    length = max_question_len

  for j in range(length):
    tmp_list=question[i].split()
    if tmp_list[j] in unique_word_index.keys():
      val_question_arr[i][j] = unique_word_index[tmp_list[j]]
    else:
      val_question_arr[i][j] = 0

x_tr = train_answer_arr
x_val = val_answer_arr
y_tr = train_question_arr
y_val = val_question_arr

train_answer_arr=0
val_answer_arr=0
train_question_arr=0
val_question_arr=0
answer=0
train_answer_list=0
question=0
train_question_list=0

print("Size of vocabulary from the w2v model = {}".format(num_tokens))

K.clear_session()

latent_dim = 512
embedding_dim = 300

# Encoder
#encoder_inputs = Input(shape=(), dtype=tf.string, name="enc_inputs")

#embedding layer
encoder_inputs = Input(shape=(None,),name='enc_inputs')

#embedding layer
enc_emb_layer =  Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    name='enc_emb'
)

enc_emb = enc_emb_layer(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,name='enc_lstm1')
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,),name='dec_inputs')

#embedding layer
dec_emb_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    name='dec_emb'
)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm_0 = LSTM(latent_dim, return_sequences=True, return_state=True, name='dec_lstm_0')
decoder_outputs_0,decoder_fwd_state, decoder_back_state = decoder_lstm_0(dec_emb,initial_state=[state_h1, state_c1])

#dense layer
decoder_dense =  TimeDistributed(Dense(num_tokens, activation='softmax',name='dense_outputs'))
decoder_outputs = decoder_dense(decoder_outputs_0)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='loss', mode='auto', verbose=1,patience=3)

checkpoint_filepath = 'best_model.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='auto',
    save_best_only=True)

#model.load_weights('best_model.h5')
history=model.fit([x_tr,y_tr[:,:-1]],
                  y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:],
                  epochs=1,
                  callbacks=[es, model_checkpoint_callback],
                  batch_size=4)

key_list = list(unique_word_index.keys())
val_list = list(unique_word_index.values())

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_output1, state_h1, state_c1])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_answer_len,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs0, state_h2, state_c2 = decoder_lstm_0(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs0)

# Final decoder model
inp = [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c]
out = [decoder_outputs2] + [state_h2, state_c2]
decoder_model = Model(
    inp, out
    )

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = unique_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = key_list[sampled_token_index]

        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_question_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=unique_word_index['sostok']) and i!=unique_word_index['eostok']):
            newString=newString+key_list[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+key_list[i]+' '
    return newString

test_set_answer=[]
test_set_que_predicted=[]
test_set_que_original=[]

for i in tqdm(range(len(x_val))):
    test_set_answer.append(x_val[i])
    test_set_que_original.append(seq2summary(y_val[i]))
    string = [str(x_val[i])]
    test_set_que_predicted.append((decode_sequence(string).split('start ')[1]).split(' end')[0])

df = pd.DataFrame(list(zip(test_set_que_predicted, test_set_que_original)),
               columns =['predicted', 'real'])

from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from evaluate import load
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import evaluate
#bertscore = load("bertscore")
meteor = evaluate.load('meteor')

rouge = Rouge()
bert_scores=0
one_gram=0
two_gram=0
three_gram=0
four_gram=0
rouge_1=0
rouge_2=0
rouge_l=0
meteor_score=0

for i in range(len(df)):
  predictions = df['predicted'][i]
  references  = df['real'][i]

  #for bert score
  #bert_scores = bert_scores+ bertscore.compute(predictions=[predictions], references=[references], model_type="distilbert-base-uncased")['f1'][0]


  #for BLEU score
  one_gram = one_gram + sentence_bleu([references.split()], predictions.split(), weights=(1, 0, 0, 0))
  two_gram = two_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.5, 0.5, 0, 0))
  three_gram = three_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.33, 0.33, 0.33, 0))
  four_gram = four_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.25, 0.25, 0.25, 0.25))

  #for ROUGE score
  rouge_scores = rouge.get_scores(predictions, references)
  rouge_1 = rouge_1 + rouge_scores[0]['rouge-1']['f']
  rouge_2 = rouge_2 + rouge_scores[0]['rouge-2']['f']
  rouge_l = rouge_l + rouge_scores[0]['rouge-l']['f']

  #for METEOR score
  meteor_score = meteor_score + meteor.compute(predictions=[predictions], references=[references])['meteor']

#print('bert_scores : ', bert_scores/len(df))
print('BLEU 1      : ',one_gram/len(df))
print('BLEU 2      : ',two_gram/len(df))
print('BLEU 3      : ', three_gram/len(df))
print('BLEU 4      : ', four_gram/len(df))
print('ROUGE 1     : ', rouge_1/len(df))
print('ROUGE 2     : ', rouge_2/len(df))
print('ROUGE L     : ', rouge_l/len(df))
print('METEOR      : ', meteor_score/len(df))

df = pd.DataFrame(list(zip(test_set_answer, test_set_que_original, test_set_que_predicted)),
               columns =['answer', 'real', 'predicted'])

df.to_csv('generated.csv', index=False)