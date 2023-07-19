'''
Script to aid in longitudinal data analysis

input: single folder containing each partition named '6', '7', '8', etc
1.) create the 5 minute crops from 5 - 10 minute mark of each video in each partition
2.) ensure audio is in the correct format
3.) run model across all instances

--- note: fma stands for 'five minute audio' ---
--- note: thirma stands for 'thiry minute audio' ---
---       these scripts MUST be run from outer directory ---
'''

import os
import argparse
import pandas as pd

def get_session_map(file):
    dtype = {"session-id": str, "participant-ID": str}
    df = pd.read_csv(file, header=0, dtype=dtype, keep_default_na=False)
    session_mapping = {}
    for index, row in df.iterrows():
        try:
            int_id = int(row['participant-ID'])
        except:
            continue
        session_mapping[row['session-id']] = int_id
    return session_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("seedlings_dir", help="Seedlings directory containing partitions")
    parser.add_argument("--gpu", help="Utilize a GPU for calculations", action="store_true")
    args = parser.parse_args()
    seedlings_dir = args.seedlings_dir
    gpu = args.gpu

    ## -- input checking -- ##
    if not os.path.isdir(seedlings_dir):
        raise Exception("Please enter a valid Seedlings directory")

    ## -- expect format of numbers within parent Seelings dir -- ##
    for partition_dir in os.listdir(seedlings_dir):
        # get spreadsheet data
        if partition_dir[0] == '.': continue
        if partition_dir == '6': continue
        if partition_dir == '7': continue
        if partition_dir == '8': continue
        if partition_dir == '9': continue
        if partition_dir == '10': continue
        if partition_dir == '11': continue
        if partition_dir == '12': continue
        # if partition_dir == '13': continue
        spreadsheet = os.path.join(seedlings_dir, partition_dir, 'spreadsheet.csv')
        session_map = get_session_map(spreadsheet)
        # month_7_complete = [20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46]
        for session in os.listdir(os.path.join(seedlings_dir, partition_dir)):
            if session == 'spreadsheet.csv': continue
            print('partition: ', partition_dir)
            # print('session: ', session)
            split_sess = session.split('-')
            # session_num = 0
            rel_session_name = split_sess[0]
            # if len(split_sess) > 1:
                # session_num = int(split_sess[1])

            ## -- enter in checkpoint info here for continuing -- ##
            #if partition_dir == '9' and session_num > 0 and session_num < 34: continue
            if rel_session_name in session_map.keys():
                session_num = int(session_map[rel_session_name])
                print('session: ', session_num)
                # if (partition_dir == '7' and session_num > 5 and session_num < 16) or (partition_dir == '7' and session_num in month_7_complete): continue
                # elif (partition_dir == '8' and session_num < 13) or (partition_dir == '8' and session_num > 36): continue
                if partition_dir == '13' and session_num < 37: continue

                ## --------------- change back to thirma later ----------------- ##
                audioname = 'thirma_month_' + partition_dir + '_chi_' + str(session_map[rel_session_name]) + '.wav'
                for media in os.listdir(os.path.join(seedlings_dir, partition_dir, session)):
                    if media[-4:] != '.mp4': continue
                    if 'video' not in media: continue
                    vid_path = os.path.join(seedlings_dir, partition_dir, session, media)
                    audio_output = os.path.join(seedlings_dir, partition_dir, session, audioname)
                    ## -------------- clipping 30 minute long audio -------------- ##
                    os.system("ffmpeg -y -ss 00:05:00 -t 00:30:00 -i {0} -vn -acodec pcm_s16le -ac 1 -ar 16000 {1}".format(vid_path, audio_output))
                    # -vn indicates no video, -acodec defines codec, ac defines mixdown to mono channel, ar sets sample rate

                    ## -- process the newly cropped audio -- ##
                    os.system('python process_audio.py ' + audio_output)
