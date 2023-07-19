import librosa
import os
import sys
import math
import random

eval_audio_dir = "/scratch/users/choward8/seedlings_audio/eval_audio/"
months_of_interest = ['15', '16', '17']
children = ['01', '02', '04', '06', '07', '08', '09', '11', '15', '17', '20', '22', '23', '24',
'26', '27', '30', '31', '33', '34', '37', '39', '40', '41', '42', '43', '45']
base_folder = '/scratch/groups/syyeung/seedlings/'
for month in months_of_interest:
    ##cycle through kids
    month_folder = 'month' + month
    for child in children:
        suffix = child + '_' + month + '_audio.wav'
        audio_fn = os.path.join(base_folder, month_folder, suffix)
        if not os.path.exists(audio_fn):
            print('Could not find audio file for: ', suffix)
            continue

        #set range where we find start time for random selection of audio
        audio_dur = math.floor(librosa.get_duration(filename=audio_fn))
        if audio_dur < 70:
            print('Audio file has too short a length: ', suffix)
            continue
        elif audio_dur < 300:
            start = 0
        else:
            start = 230
        end = audio_dur - 60
        start_time = random.randrange(int(start), int(end))
        output_filename = os.path.join(eval_audio_dir, "month{0}_chi_{1}.wav".format(month, child))
        os.system("ffmpeg -ss {0} -t 60 -i {1} {2}".format(start_time, audio_fn, output_filename))
