import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.layers import *
import tensorflow_hub as hub
from keras.models import Model
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel,Seq2SeqArgs
import json
import torch
import random
import os
from nltk.translate.meteor_score import meteor_score
from evaluate import load
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import evaluate
bertscore = load("bertscore")
meteor = evaluate.load('meteor')
random.seed(42)

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)

# Initialize model

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 60
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.overwrite_output_dir = True

model = Seq2SeqModel(encoder_decoder_type="bart", 
                     encoder_decoder_name="facebook/bart-large", 
                     args=model_args, 
                     use_cuda=True, 
                     max_length=50, 
                     batch_size=32,
                     early_stopping = True,
                     num_beams = 5
                     )


# Load & split the data

filename = "data/version_3.json"
data = pd.read_json(filename)

source_column = "concated_transcript"
target_column = "mmcs" # For MCS, target_column = "One_line_patient_summary"

df = data
x_trn, x_test, y_train, y_test = train_test_split(df,
                                                  df[target_column],
                                                test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_test, 
                                                x_test[target_column],
                                                test_size=0.7)

trn_df = x_trn[[source_column, target_column]]
val_df = x_val[[source_column, target_column]]
test_df = x_test[[source_column, target_column]]


trn_df.columns=['input_text', 'target_text']
val_df.columns=['input_text', 'target_text']
test_df.columns=['input_text', 'target_text']

# Train the model

model.train_model(trn_df, eval_data=val_df)


# Generate predictions

test_df = test_df.reset_index()
generated=[]
actual=[]
patient_dialogue=[]
for i in range(len(test_df)):
  actual.append(test_df['target_text'][i])
  patient_dialogue.append(test_df['input_text'][i])
  pred = model.predict([test_df['input_text'][i]])[0]

  generated.append(pred)

new = pd.DataFrame(list(zip(patient_dialogue, actual, generated)),
               columns =['Input text', 'Actual', 'Generated'])

# Calculate metrics & display

rouge = Rouge()
bert_scores=0
one_gram=0
two_gram=0
three_gram=0
four_gram=0
rouge_1=0
rouge_2=0
rouge_l=0
J = 0

meteor_score=0
weights_1 = (1./1.,)
weights_2 = (1./2. , 1./2.)
weights_3 = (1./3., 1./3., 1./3.)
weights_4 = (1./4., 1./4., 1./4., 1./4.)


df = new

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


for i in range(len(df)):
  predictions = df['Generated'][i]
  references  = df['Actual'][i].replace('\n', '')

  #for bert score
  bert_scores = bert_scores+ bertscore.compute(predictions=[predictions], references=[references], model_type="distilbert-base-uncased")['f1'][0]


  #for BLEU score
  one_gram = one_gram + sentence_bleu([references.split()], predictions.split(), weights=(1, 0, 0, 0))
  two_gram = two_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.5, 0.5, 0, 0))
  three_gram = three_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.33, 0.33, 0.33, 0))
  four_gram = four_gram + sentence_bleu([references.split()], predictions.split(), weights=(0.25, 0.25, 0.25, 0.25))

  J += jaccard(references.split(), predictions.split())


  #for ROUGE score
  if references == '.':
    references='empty'
  rouge_scores = rouge.get_scores(predictions, references)
  rouge_1 = rouge_1 + rouge_scores[0]['rouge-1']['f']
  rouge_2 = rouge_2 + rouge_scores[0]['rouge-2']['f']
  rouge_l = rouge_l + rouge_scores[0]['rouge-l']['f']

  #for METEOR score
  meteor_score = meteor_score + meteor.compute(predictions=[predictions], references=[references])['meteor']

#print('bert_scores : ', bert_scores/len(df))
print('BLEU 1      : ', one_gram/len(df))
print('BLEU 2      : ', two_gram/len(df))
print('BLEU 3      : ', three_gram/len(df))
print('BLEU 4      : ', four_gram/len(df))
print('ROUGE 1     : ', rouge_1/len(df))
print('ROUGE 2     : ', rouge_2/len(df))
print('ROUGE L     : ', rouge_l/len(df))
print('METEOR      : ', meteor_score/len(df))
print('BertScore      : ', bert_scores/len(df))
print('Jaccard Score      : ', J/len(df))

import pickle
scores = {'BLEU_1' : one_gram/len(df),
          'BLEU_2' : two_gram/len(df),
          'BLEU_3' : three_gram/len(df),
          'BLEU_4' : four_gram/len(df),
          'ROUGE_1' : rouge_1/len(df),
          'ROUGE_2' : rouge_2/len(df),
          'ROUGE_L' : rouge_l/len(df),
          'METEOR' : meteor_score/len(df),
          'Bert_Score' : bert_scores/len(df),
          'Jaccard Score' : J/len(df)}

# Save predictions & metrics

score_filename = "BART_OS.p"
pickle.dump(scores, open(score_filename, 'wb'))

pred_filename = "BART_OS.csv"
new.to_csv('BART_OS.csv', index=False)
