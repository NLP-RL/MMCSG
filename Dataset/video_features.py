import pandas as pd
from Katna.video import Video
import os
import cv2
import os
from pytube import YouTube
from datetime import timedelta
import numpy as np
from tqdm import tqdm 

df = pd.read_csv('data/dataset_3.csv')

text = list(df['Video Link'])

# %% parameters

FRAMES_PER_SECOND = 10  # for 10 frame per second

set_resolution = 720  # works only for 360 and 720



_savevideo = "YoutubeVideo"  # make a directory for video download
_saveframes = "ExtractedImageFrames"

# %%
isExist = os.path.exists(_savevideo)
if not isExist:
    os.makedirs(_savevideo)


# %%

def Download(link):
    """
    parameters:
        link: path for youtube video to be downloaded
    """

    youtubeObject = YouTube(link)
    original_resolution = (youtubeObject.streams.get_highest_resolution().resolution)
    strip_resolution = int(original_resolution[:-1])
    #print((strip_resolution))
    if (strip_resolution < set_resolution):
        select_resolution = (strip_resolution)
        chosen_resolution = str(select_resolution) + "p"
    else:
        chosen_resolution = str(set_resolution) + "p"
    #print(chosen_resolution)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    # youtubeObject = youtubeObject.streams.filter(res=chosen_resolution).first()
    path = 'YoutubeVideo/' + link.split('watch?v=')[1]
    youtubeObject.download(output_path=path)

    #print("Download is completed for {}".format(link))


# %%

# Path for youtube url
myPath = "file.txt"

# with open(text, 'r') as f:
#    links_1 = f.readlines()

links_1 = [k.rstrip('\n') for k in text]
#print(text[:10])
downloadedvideo = os.listdir("./YoutubeVideo")
#print(len(downloadedvideo))

links = list(set(links_1) - set(downloadedvideo))
#print(links[:10])

#print("left video is {}".format(len(links)))

# %%
notFoundURL = []  # store urls not found

# os.chdir(_savevideo) # change directory to save youtuvevideo

for link in tqdm(text):
    link_temp = link.strip()
    try:
        Download(link_temp)
    except:
        notFoundURL.append(link_temp)
        #print("An error has occurred")
        continue

print("Not found urls {}".format(notFoundURL))
# # %%
# # save videos not found

urlNotFound = pd.DataFrame()
urlNotFound['urls'] = notFoundURL
urlNotFound.to_csv('URLNotFound.csv')

# %%
def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    if cap.get(cv2.CAP_PROP_FPS) == 0:
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


# %%

def save_frames(link):
    """
    parameters:
        link: path for youtuve video
        output: saves all the frames
    """
    cam = cv2.VideoCapture(_savevideo + '//' + link)
    # get fps information
    fps = cam.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = max(fps, FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cam, saving_frames_per_second)
    filename = _saveframes + '//' + link
    if not os.path.exists(filename):
        os.makedirs(filename)
    # start the loop
    count = 0
    while True:
        is_read, frame = cam.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration,
            # then save the frame
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame)
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1
    cam.release()
    cv2.destroyAllWindows()
    return fps


# %%
# os.chdir(_savevideo)
listYoutubevideo = os.listdir(_savevideo)

if not os.path.exists(_saveframes):
    os.makedirs(_saveframes)

fps_all = []
video_name = []


for link in listYoutubevideo:
    # print(link)
    fps = save_frames(link)
    fps_all.append(fps)
    video_name.append(link)

df = pd.DataFrame()
df['Video name'] = video_name
df['original rate information'] = fps_all
df.to_csv(".//FPS.xlsx")

videopath = "YoutubeVideo/"                         # path for video
output_folder_video_image = "ExtractedImageFrames"  # path for saving images
noImages = 10
videofiles = os.listdir(videopath)
vd = Video()

for file in tqdm(videofiles):
    temp = os.listdir(videopath + file)[0]
    path = videopath + file + '/' + temp
    #print(path)
    imgs = vd.extract_video_keyframes_big_video(no_of_frames = noImages, file_path= path)
    makdir = output_folder_video_image+'/'+ file

    if not os.path.exists(makdir):
        os.mkdir(makdir)
    for i in range(noImages):
        vd.save_frame_to_disk(imgs[i], file_path=makdir, \
         file_name="Img_"+str(i), file_ext=".jpeg")

data = pd.read_csv('data/dataset_3.csv')

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


# initialize video module
vd = Video()

# print(data.video_paths[0][2:])

# print(os.path.exists(data.video_paths[0][2:]))
# print(os.listdir("Videos"))
# print(os.path.exists(os.path.join("Videos",data.video_ids[0]+".mp4")))

# number of images to be returned


no_of_frames_to_returned = 10

for i in tqdm(range(len(data))):
    # print(os.path.exists(data.video_paths[i]))
    # initialize diskwriter to save data at desired location
    if os.path.exists("ExtractedImageFrames/" + data['Video Link'][i].split('watch?v=')[1] + "/"):
        # os.mkdir("ExtractedImageFrames/"+data['Video Link'][i].split('watch?v=')[1]+"/")
        #print(os.path.exists("ExtractedImageFrames/"+data['Video Link'][i].split('watch?v=')[1]+"/"))
        diskwriter = KeyFrameDiskWriter(location="ExtractedImageFrames/"+data['Video Link'][i].split('watch?v=')[1]+"/")


        temp = os.listdir('YoutubeVideo/'+data['Video Link'][i].split('watch?v=')[1])[0]

    	# Video file path
        video_file_path = os.path.join("YoutubeVideo", data['Video Link'][i].split('watch?v=')[1], temp)

        #print(f"Input video file path = {video_file_path}")

    	# extract keyframes and process data with diskwriter
        vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
        writer=diskwriter
    )

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os
import json



_saveframes = "ExtractedImageFrames" 

device = torch.cuda.is_available()
# Loading the Model
resnet152 = models.resnet152(pretrained=True)
resnet152.fc = nn.Identity() # Removing the last layer


# Feature Extraction 
feature_fr_vid = []
seq_feature_fr_vid = []
vid_ids=[]

for i in tqdm(range(len(data))):

    path = "ExtractedImageFrames/"+ data['Video Link'][i].split('watch?v=')[1] +"/"
    feat = []
    if os.path.exists("ExtractedImageFrames/"+ data['Video Link'][i].split('watch?v=')[1] +"/"):
       for j in os.listdir(path):
            img = cv2.imread(path+ j)
            resized_img = cv2.resize(img,(224,224))
            feature = resnet152(torch.Tensor(resized_img).permute(2,0,1).reshape(1,3,224,224))
            feat.append(feature.detach().numpy())
       #print('hello', feat)
       #print('hi   ', np.array(feat).mean(axis = 0))
       vid_ids.append(data['Video Link'][i].split('watch?v=')[1])
       feature_fr_vid.append(np.array(feat).mean(axis = 0))
       seq_feature_fr_vid.append(feat)

len(vid_ids)
df = pd.DataFrame(list(zip(vid_ids, feature_fr_vid, seq_feature_fr_vid)),
               columns =['vid_ids', 'ffv', 'sffv'])

df.to_json('new_df.json')

