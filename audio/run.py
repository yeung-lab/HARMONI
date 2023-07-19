'''
Quick util file to convert all flac files in a directory into a wav file with the following encoding (usable in Voice Type Classifier)
codec: PCM S16 LE
channel: mono
sample rate: 16khz
bits per sample: 16
wrapper: wav

Assumes the input file is a valid audio file (FLAC, wav, mp3, etc)
Will save the converted audio in the some folder
'''

import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("usage: python preprocess.py path/to/audio/file path/to/out/folder")

    audio_path = sys.argv[1]
    out_folder = sys.argv[2]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if os.path.isfile(audio_path):

        # convert to wav
        basename = os.path.basename(audio_path)
        new_file_path = os.path.join(out_folder, basename.replace(".mp4", ".wav"))
        if not os.path.exists(new_file_path):
            os.system("ffmpeg -i {} -vn -acodec pcm_s16le -ac 1 -ar 16000 {}".format(audio_path, new_file_path))

        # process audio
        os.system('python process_audio.py ' + new_file_path + " --gpu")
    else:
        raise Exception("Invalid file path", audio_path)
