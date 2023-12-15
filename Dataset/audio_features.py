import pandas as pd
from tqdm import tqdm
import pytube
from moviepy.editor import *

df = pd.read_csv('data/dataset_3.csv')

print('Number of rows ' + str(len(df)))

folder = './audio'
def audio_download(video_url):
    youtube = pytube.YouTube(video_url)
    video = youtube.streams.get_audio_only()
    video_id = video_url.split('?v=')[1]
    video.download(filename=folder + '/' + video_id + '.mp4')
    video_clip = AudioFileClip(folder + '/' + video_id + '.mp4')
    video_clip.write_audiofile(folder + '/' + video_id + '.wav')
    os.remove(folder + '/' + video_id + '.mp4')

yt_links = list(df['Video Link'])

c=0
for i in tqdm(range(len(df))):
    try:
        audio_download(df['Video Link'][i])
    except:
        c = c+1
        print("An exception occurred")

print('unable to download audio files of '+ str(c) + ' urls')

import imageio
imageio.plugins.ffmpeg.download()

import wave, math, contextlib
from moviepy.editor import AudioFileClip
import opensmile
import os
import pandas as pd

audiopath = "./audio"

audiopath = "audio"
audiofiles = os.listdir(audiopath)

smile_functional = opensmile.Smile(
feature_set=opensmile.FeatureSet.emobase,
feature_level=opensmile.FeatureLevel.Functionals
)

smile_LLD = opensmile.Smile(
feature_set=opensmile.FeatureSet.ComParE_2016,
feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)

audio_features = dict()
start = 0

for file in audiofiles: 
    print(file)
    df2 = smile_functional.process_file(audiopath+'//'+file)
    df2 = df2.mean(axis=0)
    df2 = pd.DataFrame(df2)
    df2 = df2.T
    df2 = df2.reset_index(drop=True)
    df3 = smile_LLD.process_file(audiopath+'//'+file)
    df3 = df3.mean(axis=0)
    df3 = pd.DataFrame(df3)
    df3 = df3.T
    df3 = df3.reset_index(drop=True)
    df_temp = pd.concat([df2,df3], axis=1)
    print("completed file {}".format(file))
    #df_temp.index.rename(file,inplace=True)
    df_temp['Video_name'] = file
    print(df_temp.index)
    if(start==0):
        df_final = df_temp
        start = start+1
    else:
        df_final = pd.concat([df_final,df_temp],axis=0)
df_final.reset_index(drop=True, inplace = True)

# shift column 'Name' to first position
first_column = df_final.pop('Video_name')
df_final.insert(0, 'Video_name', first_column)

df_final.to_json('Audio_features.json')
df_final.to_csv('Audio_features.csv')
