import json
import pandas as pd
import transformers
import torch
import numpy as np
import sklearn

# Load BERT model
model = transformers.BertModel.from_pretrained('bert-base-uncased')
tokeniser = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv('data/dataset_3.csv')
df = df.reset_index()
df = df.drop(['index', ' Sl No.', 'Durration '], axis=1)

temp_dict = {}
for i in range(len(df)):
    temp_dict[df['Video Link'][i].split('?v=')[1]] = df['Video Title'][i]

df.rename(columns = {'Video Link':'url',
                     'Video Title':'title',
                     'Patient Gender':'patient_gender',
                     'Patient Age Group':'patient_age_group',
                     'Transcript':'transcript',
                     'Subtitle':'utterances',
                     'Focus Point (Patient_Video)':'focal_utterances',
                     'Doctor Suggestion Summary':'dcs',
                     'Patient Concern Summary':'mmcs',
                     'Intent 1':'intent'}, inplace = True)

df['patient_age_group'] = df['patient_age_group'].astype(str)
df['patient_gender'] = df['patient_gender'].astype(str)

df = df.astype(str)

intent_dict= {
    'Cause':'0',
    'Information':'1',
    'Suggestion':'2',
    'Treatment':'2',
    'Affect':'2',
    'Prevention':'2',
    'Test':'2',
    'nan':'2'
}

temp_list_1=[]
temp_list_2=[]
temp_list_3=[]
for i in range(len(df)):
    temp = intent_dict[df['intent'][i]]
    temp_list_1.append(temp)
    temp_list_2.append(df['mmcs'][i]+'<EXP><0,0>')

df['encoded_merged_intent'] = temp_list_1
df['final_mmcs'] = temp_list_2
df['merged_intent'] = df['intent']

for i in range(len(df)):
    if df['merged_intent'][i] == 'Treatment':
        df['merged_intent'][i] = 'Suggestion'
    if df['merged_intent'][i] == 'Affect':
        df['merged_intent'][i] = 'Suggestion'

personality_info = "<GEN> "+df.patient_gender+" <AGE> " + df.patient_age_group

# encoding personlity features(BERT encoding)
personality_features = []

for i in personality_info:
    input_ids = torch.tensor([tokeniser.encode(i,max_length=14,padding="max_length")])
    features = model(input_ids)[0]
    #print(features.shape)
    personality_features.append(features.detach().numpy())

df["personality_features"] = personality_features
df["personality_features"].apply(lambda x:x.shape[1]).max()

from sklearn.preprocessing import LabelEncoder
values = np.array(df.merged_intent)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

df["encoded_merged_intent"] = integer_encoded

# video features json file
json_file_path = "video_features.json"

with open(json_file_path, 'r') as j:
    video_features = json.loads(j.read())

# audio feature json file
json_file_path = "Audio_features.json"

with open(json_file_path, 'r') as j:
    audio_features = json.loads(j.read())

json_file_path = 'sample.json'

with open(json_file_path, 'r') as j:
    old_file = json.loads(j.read())

string=[]
for i in range(len(df)):
    df['transcript'][i] = str(df['transcript'][i]).replace('\r', '')
    temp_list = str(df['transcript'][i]).split('\n\n')
    temp_string=''
    for j in range(len(temp_list)):
        temp_string = temp_string + str(j+1) + ', ' + temp_list[j] + ' '
    temp_list=[]
    string.append(temp_string)

df['concated_transcript'] = string

id_list=[]
for i in range(len(df)):
    id = df['url'][i].split('?v=')[1]
    id_list.append(id)

temp_list=[]
video_features['vid_ids']['0']
for i in range(len(video_features['vid_ids'])):
    temp_list.append(video_features['vid_ids'][str(i)])

id_title={}
for i in range(len(df)):
    id = df['url'][i].split('v=')[1]
    id_title[id] = df['title'][i]

ad = sorted(list(audio_features['Video_name'].values()))
ad = [x.split('.wav')[0] for x in ad]

updated_video_features={}
for i in range(len(video_features['vid_ids'].keys())):
    updated_video_features[video_features['vid_ids'][str(i)]] = video_features['ffv'][str(i)][0]

title_int = audio_features['Video_name']
for i in title_int.keys():
    title_int[i] = title_int[i].split('.wav')[0]

# Removing audio featuers with only 0 values for every audio
temp_list=[]

# audio feature json file
json_file_path = "Audio_features_version_2.json"

with open(json_file_path, 'r') as j:
    audio_features_version_2 = json.loads(j.read())


for i in audio_features_version_2.keys():
    sum=0
    for j in range(len(audio_features_version_2['Video_name'].keys())):
        if audio_features_version_2[i][str(j)] == 0:
            sum=sum+1
    if sum == len(audio_features_version_2['Video_name'].keys()):
            temp_list.append(i)

for i in temp_list:
    audio_features.pop(i)

temp_file = audio_features
length_rows = len(temp_file['Video_name'].keys())
temp_file.pop('Video_name')

temp_list=[]

updated_audio_features={}
for i in range(length_rows):
    temp_list=[]
    for key in list(temp_file.keys()):
        a = temp_file[key][str(i)]
        temp_list.append(a)
    updated_audio_features[title_int[str(i)]] = temp_list

url_list=[]
title_list=[]
transcript_list=[]
patient_age_group_list=[]
patient_gender_list=[]
mmcs_list=[]
intent_list=[]
image_features_list=[]
audio_features_list=[]
utterances_list=[]
focal_utterances_list=[]
concated_transcript_list=[]
final_mmcs_list=[]
merged_intent_list=[]
personality_features_list=[]
encoded_merged_intent_list=[]
dcs_list=[]

for i in updated_video_features.keys():
    for j in range(len(df)):
        if i == df['url'][j].split('?v=')[1]:
            url_list.append(df['url'][j])
            title_list.append(df['title'][j])
            transcript_list.append(df['transcript'][j])
            patient_age_group_list.append(df['patient_age_group'][j])
            patient_gender_list.append(df['patient_gender'][j])
            mmcs_list.append(df['mmcs'][j])
            intent_list.append(df['intent'][j])
            image_features_list.append(updated_video_features[i])
            audio_features_list.append(updated_audio_features[i])
            utterances_list.append(df['utterances'][j])
            focal_utterances_list.append(df['focal_utterances'][j])
            concated_transcript_list.append(df['concated_transcript'][j])
            final_mmcs_list.append(df['final_mmcs'][j])
            merged_intent_list.append(df['merged_intent'][j])
            personality_features_list.append(df['personality_features'][j].tolist())
            encoded_merged_intent_list.append(float(df['encoded_merged_intent'][j]))
            dcs_list.append(df['dcs'][j])

new_df = pd.DataFrame(url_list, columns =['url'])
new_df['title'] = title_list
new_df['transcript'] = transcript_list
new_df['patient_age_group'] = patient_age_group_list
new_df['patient_gender'] = patient_gender_list
new_df['mmcs'] = mmcs_list
new_df['intent'] = intent_list
new_df['image_features'] = image_features_list
new_df['audio_features'] = audio_features_list
new_df['utterances'] = utterances_list
new_df['focal_utterances'] = focal_utterances_list
new_df['concated_transcript'] = concated_transcript_list
new_df['final_mmcs'] = final_mmcs_list
new_df['merged_intent'] = merged_intent_list
new_df['personality_features'] = personality_features_list
new_df['encoded_merged_intent'] = encoded_merged_intent_list
new_df['dcs'] = dcs_list
new_df = new_df.drop_duplicates(subset=['title'])
new_df = new_df.reset_index()

for i in range(len(new_df)):
    for j in old_file.keys():
        old_file[j][str(i+386)] = new_df[j][i]

with open("version_3.json", "w") as outfile:
    json.dump(old_file, outfile)