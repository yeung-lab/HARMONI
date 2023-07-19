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


def convert_audio(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    if not basename.endswith("_video"):
        print("invalid file found: ", basename)
        return
    basename = basename[:-6]
    output_filename = basename + '_audio.wav'
    parent_dir = os.path.dirname(filename)
    if not parent_dir:
        parent_dir = '.'
    new_file_path = os.path.join(parent_dir, output_filename)
    if os.path.exists(new_file_path):
        print("File {} already created...".format(output_filename))
        return
    os.system("ffmpeg -i {0} -vn -acodec pcm_s16le -ac 1 -ar 16000 {1}".format(filename, new_file_path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("usage: python audio_converter.py path/to/directory/with/audio/file")

    audio_path = sys.argv[1]
    if os.path.isdir(audio_path):
        for filename in os.listdir(audio_path):
            if not filename.endswith(".mp4"):
                continue ## temporary change for this particular use case
            full_file_name = os.path.join(audio_path, filename)
            convert_audio(full_file_name)
    elif os.path.isfile(audio_path):
        convert_audio(audio_path)
