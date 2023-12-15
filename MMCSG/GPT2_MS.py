import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import warnings
import json
import torch
import random
import pickle
import gc
import os
from nltk.translate.meteor_score import meteor_score
from evaluate import load
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import random
import evaluate
import pandas as pd
bertscore = load("bertscore")
meteor = evaluate.load('meteor')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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


# Load the JSON dataset
def load_dataset(file_path):
    import json
    import pandas as pd

    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Convert DataFrame back to a list of dictionaries
    data = df.to_dict(orient='records')

    return data


# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2",  model_max_length=480, max_length=480, truncation=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})   
        self.max_length = 480


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item[source_column]
        target_text = item[target_column]
        
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens=False, return_token_type_ids=False, max_length=480, truncation=True, padding='max_length')
        target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False, return_token_type_ids=False, max_length=480, truncation=True, padding='max_length')

        input_tokens = input_tokens[:self.max_length]

        input_tensor = torch.tensor(input_tokens)
        target_tensor = torch.tensor(target_tokens)

        return input_tensor, target_tensor

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_length = max(len(seq) for seq in inputs + targets)

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    # Adjust batch sizes if necessary
    if inputs_padded.size(1) != targets_padded.size(1):
        diff = targets_padded.size(1) - inputs_padded.size(1)
        if diff > 0:
            inputs_padded = torch.nn.functional.pad(inputs_padded, (0, diff), value=0)
        else:
            targets_padded = torch.nn.functional.pad(targets_padded, (0, -diff), value=0)

    return inputs_padded, targets_padded



# Define your training parameters
batch_size = 8
num_epochs =60
learning_rate = 5e-5
source_column = "concated_transcript"
target_column = "mmcs"  #For MCS, target_column = "One_line_patient_summary"

from sklearn.model_selection import train_test_split

def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data


NewDataset_path = 'data/version_3.json'
Dataset = pd.DataFrame(read_json_data(NewDataset_path))


train_data , test_data = train_test_split(Dataset,test_size = 0.2)
valid_data , test_data = train_test_split(test_data,test_size = 0.7)


train = train_data.to_json()
valid = valid_data.to_json()
test = test_data.to_json()

with open("train.json", "w") as outfile:
    outfile.write(train)

with open("val.json", "w") as outfile:
    outfile.write(valid)

with open("test.json", "w") as outfile:
    outfile.write(test)


# Load the splitted datasets
train_file = "train.json"
valid_file = "val.json"
test_file = "test.json"


train_data = load_dataset(train_file)
valid_data = load_dataset(valid_file)
test_data = load_dataset(test_file)


train_dataset = CustomDataset(train_data)
valid_dataset = CustomDataset(valid_data)
test_dataset = CustomDataset(test_data)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn = collate_fn)

print("Data_loaded")


# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


model.to(device)


# Set the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
print("Model_loaded")


# Early stopping configuration
best_valid_loss = float('inf')
patience = 3
counter = 0
best_model_path = "best_model.pth"


# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        inputs = inputs.long()

        # Forward pass
        outputs = model(inputs, labels=targets)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {average_loss}")

    # Perform validation
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs, labels=targets)
            loss = outputs.loss

            valid_loss += loss.item()

    # Calculate average validation loss
    average_valid_loss = valid_loss / len(valid_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Average Validation Loss: {average_valid_loss}")

    # Check for early stopping
    if average_valid_loss < best_valid_loss:
        best_valid_loss = average_valid_loss
        counter = 0
        # Save the best model checkpoint
        torch.save(model.state_dict(), best_model_path)
    #else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break


# Load the best model checkpoint

model.load_state_dict(torch.load(best_model_path))
model.eval()


# Generate predictions

predictions = []

with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        print(len(inputs))
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Generate predictions
        outputs = model.generate(inputs, max_length=50)  # Adjust max_length as needed
        predictions.extend(outputs)


# Convert token IDs back to text
decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
print("Predictions generated")

dec_pred = []
for i, prediction in enumerate(decoded_predictions):
    dec_pred.append(prediction)

actual=[]
patient_dialogue=[]
for i in range(len(test_data)):
  actual.append(test_data[i][target_column])
  patient_dialogue.append(test_data[i][source_column])

new = pd.DataFrame(list(zip(patient_dialogue, actual, decoded_predictions)),
               columns =['Input text', 'Actual', 'Generated'])



# Metric calculation

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

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


df = new
print(len(df))
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

score_filename = "GPT_OS.p"
pickle.dump(scores, open(score_filename, 'wb'))

pred_filename = "GPT_OS.p"
df.to_csv(pred_filename, index=False)
