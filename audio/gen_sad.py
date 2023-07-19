'''
Utility script to generate SAD .lab files from the predicted speech segments stored in the
SPEECH rttm file output by VTC --> will be consumed in the individual speaker diarization script
'''

import os
import sys
import pandas as pd


def get_rttm_df(file):
    names = [ "NA1", "uri", "NA2", "start", "duration", "NA3", "NA4", "speaker", "NA5", "NA6",]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(file, names=names, dtype=dtype, delim_whitespace=True, keep_default_na=False)
    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("usage: python speech_seg_cropper.py <speech_file>")
        ## -- example usage -- ##
        # python gen_sad.py output_voice_type_classifier/DH_0001/SPEECH.rttm

    speech_file = sys.argv[1]
    if not os.path.isfile(speech_file):
        raise Exception("Please input a file containing Speech predictions!")

    df = get_rttm_df(speech_file)
    info_to_write = [elements for elements in zip(df['start'], df['duration'])]
    file_dir, __ = os.path.split(speech_file)

    filename = df['uri'][0] + '.lab'
    fullname = os.path.join(file_dir, filename)
    with open(fullname, 'w') as writer:
        for start, dur in info_to_write:
            to_write = str(round(start, 3)) + ' ' + str(round(start + dur, 3)) + ' speech\n'
            writer.write(to_write)
