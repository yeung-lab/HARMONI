

import os
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("seedlings_month", help="Seedlings month of interest")
    parser.add_argument("--gpu", help="Utilize a GPU for calculations", action="store_true")
    args = parser.parse_args()
    seedlings_month = args.seedlings_month
    gpu = args.gpu

    ## -- input checking -- ##

    default_parent_dir = "/scratch/groups/syyeung/seedlings/"
    dir_of_interest = os.path.join(default_parent_dir, "month" + str(seedlings_month))
    if not os.path.isdir(dir_of_interest):
        raise Exception("Please enter a valid Seedlings month value")

    ## -- expect format of numbers within parent Seelings dir -- ##
    attempts = {}
    print('Starting loop!')
    still_running_files = True
    while still_running_files:

        still_running_files = False
        for file in os.listdir(dir_of_interest):
            if not file.endswith(".wav"): continue

            ##check if valid results already exists
            bn = file[:-4]
            check_file = os.path.join("output", bn, bn+".rttm")
            if os.path.isdir(check_file):
                print("Audio file {} already processed...skipping".format(bn))
                continue

            ## valid file and no result yet
            if ((file in attempts) and (attempts[file] > 3)):
                print("Reached maxed attempts for file: ", bn)
                continue
            elif file not in attempts:
                attempts[file] = 1
            else:
                attempts[file]  = attempts[file] + 1

            still_running_files = True
            audio_file = os.path.join(dir_of_interest, file)
            ## process files
            if gpu:
                os.system('python process_audio.py ' + audio_file + " --gpu")
            else:
                os.system('python process_audio.py ' + audio_file)
