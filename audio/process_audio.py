import argparse
import os
from utils import gen_ctc
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="audio file to process")
    parser.add_argument("--gpu", help="Utilize a GPU for calculations", action="store_true")
    args = parser.parse_args()
    audio = args.audio_file
    gpu = args.gpu

    if gpu: print('Using GPU!')
    else: print('Using CPU!')

    ## -- input checking -- ##
    if not os.path.exists(audio):
        print(audio)
        raise Exception("Please enter a valid audio file!")
    if len(audio) < 5 or audio[-4:] != '.wav':
        raise Exception("Please process a wav file!")
    bn = os.path.splitext(os.path.basename(audio))[0]
    print('-------- processing file ' + bn + ' -------------')
    parent_dir = os.path.dirname(audio)
    if not parent_dir:
        parent_dir = '.'

    # -- To Do: write script to verify audio characteristics of wav file --> use soxi and audio_converter.py file!

    start = time.time()

    # -- step 1: feed into VTC to generate BC rttm files -- ##
    print('---------- Step 1: generate broadclass diarizations ----------')
    if gpu:
        os.system("ALICE/voice-type-classifier/apply.sh {0} --gpu".format(audio))
    else:
        os.system("ALICE/voice-type-classifier/apply.sh {0}".format(audio))

    # step 1b: create lab file from speech.rttm file
    os.system("python gen_sad.py output/{0}/intermed/SPEECH.rttm".format(bn))
    vad_file = "output/{0}/intermed/{0}.lab".format(bn)
    bc_rttm = "output/{0}/intermed/all.rttm".format(bn)

    # -- step 2: generate individual diarization rttm file using VBx -- ##
    print('---------- Step 2: generate individual diarizations ----------')
    # note using dihard branch of VBx here
    # -- generate list_inference with file -- ##
    with open('VBx/inference_files/list_inference_{}'.format(bn), 'w') as file:
        file.write(bn+'\n')

    output_dir = 'output/' + bn
    intermed_dir = 'output/' + bn + '/intermed'
    os.system("VBx/run_recipe.sh all inference {0} {1} {2}".format(parent_dir, intermed_dir, bn))
    indi_rttm = intermed_dir + '/' +  bn + '.rttm'

    ## -- step 3: generate combined rttm from broadclass and individual diarizations -- ##
    print('---------- Step 3: combine individual and broadclass diarizations ----------')
    combo_rttm_file = 'output/{0}/{0}.rttm'.format(bn)
    os.system("python gen_combo_rttm.py {0} {1} {2}".format(indi_rttm, bc_rttm, combo_rttm_file))

    ## -- step 4: generate outputs -- ##
    print('---------- Step 4: generate outputs ----------')
    # - CTC output - #
    ctc = gen_ctc(combo_rttm_file, output_dir)

    # - ALICE word counts - #
    os.system("./ALICE/run_ALICE.sh {0} {1}".format(audio, output_dir))

    # # - clean up leftover folders - #
    # os.system('rm -rf ' + intermed_dir)

    print('total time: ', time.time() - start)
