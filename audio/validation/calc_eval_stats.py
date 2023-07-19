import os
from collections import defaultdict
import pandas as pd
import sys
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
import pyannote.metrics.errors.identification as error
from pyannote.core import notebook
import pyannote
import matplotlib
import math
from matplotlib import pyplot as plt
from pyannote.metrics.base import f_measure
import collections
import statistics
from scipy.stats import pearsonr
import argparse
import os
import pandas as pd

trans_dir_name = '/c/Users/Chris-Howard/Desktop/MARVL/validation/seedlings_audio_transcriptions/' ## windows
rttm_dir_name = '/Users/chrishoward/Desktop/Stanford/MARVL/sad_development/clipped_seedlings_eval_transcripts_wmd_correct/' ## mac -- keeping missed detections
rttm_dir_name_pyan = '/Users/chrishoward/Desktop/Stanford/MARVL/sad_development/clipped_seedlings_eval_transcripts_wmd_pyan_dev/'
rttm_dir_name_orig = '/Volumes/data_hauler/MARVL_PC_NEW/validation/seedlings_audio_transcriptions/rttms/'
male_syns = ["Adult Male", "Male Adult", "Adult Man", "Male Speaker"]
female_syns = ["Adult Female", "Female Adult", "Female Speaker"]

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def get_cleansed_label(label):
    if 'CHI' in label:
        return 'Child'
    elif 'FEM' in label:
        return 'Adult_Female'
    elif 'MAL' in label:
        return 'Adult_Male'

def get_word_count(file_lines, file):
    wc = 0
    for line in file_lines:
        line = line.strip()
        # skip past bad lines
        if line == "": continue
        if line.find(':') == -1: continue
        if line.count('END') != 2: continue

        line_post_spker = line[line.find(":")+2:]
        trans = line_post_spker.strip()[line_post_spker.find(" "):line_post_spker.find("END")]
        split = trans.strip().split()

        for word in split:
            if word.count('[') == 1 and word.count(']') == 1: continue
            # words[word] += 1
            wc += 1
    return wc

def get_pred_wc(file):
    pred_file = open(file, 'r', errors='replace')
    lines = pred_file.readlines()
    if len(lines) != 2:
        print('invalid file: skipping ', file)
        return 0
    pred_line = lines[1]
    return int(pred_line.split()[3])

def calc_avg_word_err():
    ## get word counts
    # words = collections.defaultdict(int)
    mean_abs_error = 0
    wcs = []
    pred_wcs = []
    all_err = []
    avg_wc = 0
    cnt = 0
    for file in os.listdir(trans_dir_name):
        if not file.endswith('txt'): continue
        transcription_file = open(os.path.join(trans_dir_name, file), 'r', errors='replace')
        lines = transcription_file.readlines()
        wc = get_word_count(lines, file)
        if wc == 0: continue

        ## get predicted wc
        month_chi = file[:-4]
        if not os.path.exists(os.path.join('/c/Users/Chris-Howard/Desktop/MARVL/validation/audio_val_model_predictions', month_chi, 'ALICE_output.txt')):
            continue
        prediction_file = os.path.join('/c/Users/Chris-Howard/Desktop/MARVL/validation/audio_val_model_predictions', month_chi, 'ALICE_output.txt')
        pred_wc = get_pred_wc(prediction_file)

        ## calculate pearsons correlation
        wcs.append(wc)
        pred_wcs.append(pred_wc)

        err_pct = abs(wc - pred_wc) / wc
        # if err >= 100:
        #     print(file, 'actual wc: ', wc, ' -- pred wc: ', pred_wc)
        all_err.append(err_pct)

        # mean_abs_error += err
        # avg_wc += wc
        # cnt += 1

    # print('mean abs wc error: ', mean_abs_error / float(cnt))
    # print('avg wc: ', avg_wc / float(cnt))
    # all_err.sort()
    print('all err: ', statistics.median(all_err))
    # calculate Pearson's correlation
    corr, _ = pearsonr(pred_wcs, wcs)
    print('Pearsons correlation: %.3f' % corr)


## -- generate rttm files for GT transcriptions -- ##
def generate_rttm_file(file_lines, filename):
    rttm_filename = filename[:-4] + '.rttm'
    rttm_filepath = os.path.join('/c/Users/Chris-Howard/Desktop/MARVL/validation/seedlings_audio_transcriptions/rttms/', rttm_filename)
    with open(rttm_filepath, 'w') as rttm_file:
        for line in file_lines:
            line = line.strip()
            # skip past bad lines
            if line == "": continue
            if line.find(':') == -1: continue

            ## get speaker2
            spker = line[:line.find(":")]
            if spker in male_syns:
                spker = 'Adult_Male'
            elif spker in female_syns:
                spker = "Adult_Female"
            elif spker == "Child":
                spker = "Child"
            else:
                continue

            print('file: ', filename)
            ## get start
            line_post_spker = line[line.find(":")+2:]
            start = line_post_spker.strip()[:line_post_spker.find(" ")]
            if len(start.split(':')) == 1: continue
            start = get_sec(start)
            print('start: ', start)

            ## get duration
            remain_line = line[line.find("END"):]
            end = remain_line[3:15]
            if len(end.split(':')) == 1: continue
            end = get_sec(end)
            dur = abs(round(end - start, 3))
            ## sometimes times are flipflopped
            if start > end:
                start = end
            # print('dur: ', dur)

            str = "SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>\n".format(filename[:-4], start, dur, spker)
            rttm_file.write(str)


def get_gt_rttm_annotation(file):
    names = ["NA1","uri","NA2","start","duration","NA3","NA4","speaker","NA5","NA6",]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(
            file,
            names=names,
            dtype=dtype,
            delim_whitespace=True,
            keep_default_na=False,
        )
    reference = Annotation()
    for index, row in df.iterrows():
        # if row['duration'] < 1: continue
        start = row['start']
        end = start + row['duration']
        label = row['speaker']
        # if label != "Adult_Female": continue
        reference[Segment(start, end)] = label
    return reference


def get_pred_rttm_annotation(file):
    names = ["NA1","uri","NA2","start","duration","NA3","NA4","speaker","NA5","NA6",]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(
            file,
            names=names,
            dtype=dtype,
            delim_whitespace=True,
            keep_default_na=False,
        )
    hypothesis = Annotation()
    for index, row in df.iterrows():
        # if row['duration'] < 1: continue
        start = row['start']
        end = start + row['duration']
        label = get_cleansed_label(row['speaker'])
        if label == 'SPEECH': continue
        # if label != "Adult_Female": continue
        hypothesis[Segment(start, end)] = label
    return hypothesis


## calculate DER
def calculate_DER():
    # subset = [30, 26, 31, 4]
    # tv = ['6_26', '7_4', '7_30', '8_26', '7_26', '8_31', '9_26', '9_30', '10_4', '10_26', '11_26', '12_31', '13_26', '13_30', '13_31', '14_31']
    num_predictions = 0
    avg_components = defaultdict(int)
    num_skipped = 0
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    valids = []
    for file in os.listdir(rttm_dir_name_pyan):
        if file[0] == '.': continue
        if not file.endswith('rttm'): continue

        anno_rttm_file = os.path.join(rttm_dir_name_pyan, file)
        # print('anno_rttm_file: ', anno_rttm_file)
        gt_labels = get_gt_rttm_annotation(anno_rttm_file)

        ## -- get prediction rttm -- ##
        month_chi = file[:-5]
        end = month_chi.find('_')
        month = month_chi[5:end]
        # checker = month + '_' + month_chi[-2:]
        # if checker in tv: continue
        ## subsample logic
        # if int(month_chi[-2:]) not in subset:
            # continue

        pred_dir = '/Volumes/data_hauler/MARVL_PC_NEW/validation/audio_val_model_predictions'
        if not os.path.exists(os.path.join(pred_dir, month_chi, file)):
            continue
        pred_rttm_file = os.path.join(pred_dir, month_chi, file)
        pred_labels = get_pred_rttm_annotation(pred_rttm_file)
        test_metric = DiarizationErrorRate(collar=0.5)
        components = test_metric(gt_labels, pred_labels, detailed=True)
        # if file == 'month10_chi_22.rttm':
        #     print('------------DER for ', file, ': ', components, '--------------')
        # if components['false alarm'] >= 6:
        #     print('DER for ', file, ': ', components['diarization error rate'])
        #     print(components)
        #     num_skipped += 1
        #     continue

        # if components['missed detection'] >= 20:
        #     print('DER for ', file, ': ', components['diarization error rate'])
        #     print(components)
        #     num_skipped += 1
        #     continue

        metric(gt_labels, pred_labels)


        for key in components:
            avg_components[key] += components[key]
        num_predictions += 1

    for key in components:
        avg_components[key] = avg_components[key] / float(num_predictions)
    # print('average results over ', num_predictions, ' samples: ', avg_components)
    print('num skipped: ', num_skipped)
    print('num predictions: ', num_predictions)
    print('total DER: ', abs(metric))
    print('componenets: ', metric[:])
    # mean, (lower, upper) = metric.confidence_interval()
    # print('mean: ', mean, lower, upper)

    # os.system("python dscore/score.py --collar 0.5 -r " + anno_rttm_file + ' -s ' + pred_rttm_file)

def calculate_diff_in_DER():
    num_predictions = 0
    avg_components = defaultdict(int)
    num_skipped = 0
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    orig_der = {}
    for file in os.listdir(rttm_dir_name_orig):
        if file[0] == '.': continue
        if not file.endswith('rttm'): continue

        anno_rttm_file = os.path.join(rttm_dir_name_orig, file)
        # print('anno_rttm_file: ', anno_rttm_file)
        gt_labels = get_gt_rttm_annotation(anno_rttm_file)

        ## -- get prediction rttm -- ##
        month_chi = file[:-5]
        end = month_chi.find('_')
        month = month_chi[5:end]


        pred_dir = '/Volumes/data_hauler/MARVL_PC_NEW/validation/audio_val_model_predictions'
        if not os.path.exists(os.path.join(pred_dir, month_chi, file)):
            continue
        pred_rttm_file = os.path.join(pred_dir, month_chi, file)
        pred_labels = get_pred_rttm_annotation(pred_rttm_file)
        test_metric = DiarizationErrorRate(collar=0.5)
        components = test_metric(gt_labels, pred_labels, detailed=True)
        metric(gt_labels, pred_labels)
        orig_der[month_chi] = components['diarization error rate']
    print('orig transcription metric: ', abs(metric))

    new_der = {}
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    for file in os.listdir(rttm_dir_name):
        if file[0] == '.': continue
        if not file.endswith('rttm'): continue

        anno_rttm_file = os.path.join(rttm_dir_name, file)
        # print('anno_rttm_file: ', anno_rttm_file)
        gt_labels = get_gt_rttm_annotation(anno_rttm_file)

        ## -- get prediction rttm -- ##
        month_chi = file[:-5]
        end = month_chi.find('_')
        month = month_chi[5:end]


        pred_dir = '/Volumes/data_hauler/MARVL_PC_NEW/validation/audio_val_model_predictions'
        if not os.path.exists(os.path.join(pred_dir, month_chi, file)):
            continue
        pred_rttm_file = os.path.join(pred_dir, month_chi, file)
        pred_labels = get_pred_rttm_annotation(pred_rttm_file)
        test_metric = DiarizationErrorRate(collar=0.5)
        components = test_metric(gt_labels, pred_labels, detailed=True)
        metric(gt_labels, pred_labels)
        new_der[month_chi] = components['diarization error rate']
    print('new transcription metric: ', abs(metric))

    diff_list = [(key, new_der[key] - val_orig) for key, val_orig in orig_der.items()]
    diff_list.sort(key=lambda x:x[1])
    print('biggest improvement: ', diff_list[:5])
    print('biggest harm: ', diff_list[-5:])


def get_indi_annotation(file):
    names = ["NA1","uri","NA2","start","duration","NA3","NA4","speaker","NA5","NA6"]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(file, names=names, dtype=dtype, delim_whitespace=True, keep_default_na=False)
    ## might want to include URI in Annotation()
    indi = Annotation()
    for index, row in df.iterrows():
        start = row['start']
        # if row['duration'] < 0.5: continue
        end = start + row['duration']
        label = row['speaker']
        indi[Segment(start, end)] = label
    return indi

def validate_ctc_segs(init_seg, resp_seg):
    if resp_seg.start < init_seg.start: return False
    ## -- introduce requirement to have initiation utterance be longer than 0.5 seconds as proxy for 'word' ensurance in LENA -- ##
    if init_seg.duration < 0.5: return False

    ## overlap case:
    if init_seg.end > resp_seg.start:
        if init_seg.end > resp_seg.end: return False
        overlap_seg = Segment(start=resp_seg.start , end=init_seg.end)
        if (overlap_seg.end - overlap_seg.start) < 5:
            return True
    ## no overlap:
    elif (resp_seg.start - init_seg.end) < 5:
        return True

    return False


def gen_ctc_pred(rttm_file):
    if not os.path.exists(rttm_file):
        raise Exception("Please enter a valid rttm file!")
    ctc = 0
    initiator_seg = None
    init_label = ''
    pred_labels = get_indi_annotation(rttm_file)
    for segment, __, label in pred_labels.itertracks(yield_label=True):
        ## need response segment to END after intiator segment --> only count KCHI in CTC
        if 'MAL' in label or 'FEM' in label:
            if initiator_seg != None and init_label == 'KCHI_1':
                if validate_ctc_segs(initiator_seg, segment):
                    ctc += 1
                    initiator_seg = None
                    init_label = ''
            else: ## curr label is male or fem, but either no initiator or initiator isn't key child
                initiator_seg = segment
                init_label = label
        elif label == 'KCHI_1':
            if initiator_seg != None and ('MAL' in init_label or 'FEM' in init_label):
                if validate_ctc_segs(initiator_seg, segment):
                    ctc += 1
                    initiator_seg = None
                    init_label = ''
            else: ## curr label is KCHI, but either no initiator or initiator isn't FEM or MAL
                initiator_seg = segment
                init_label = label
    return ctc


def gen_ctc_anno(rttm_file):
    if not os.path.exists(rttm_file):
        raise Exception("Please enter a valid rttm file!")
    ctc = 0
    initiator_seg = None
    init_label = ''
    pred_labels = get_indi_annotation(rttm_file)
    for segment, __, label in pred_labels.itertracks(yield_label=True):
        ## need response segment to END after intiator segment --> only count KCHI in CTC
        if 'Adult_Female' in label or 'Adult_Male' in label:
            if initiator_seg != None and init_label == 'Child':
                if validate_ctc_segs(initiator_seg, segment):
                    ctc += 1
                    initiator_seg = None
                    init_label = ''
            else: ## curr label is male or fem, but either no initiator or initiator isn't key child
                initiator_seg = segment
                init_label = label
        elif label == 'Child':
            if initiator_seg != None and ('Adult_Male' in init_label or 'Adult_Female' in init_label):
                if validate_ctc_segs(initiator_seg, segment):
                    ctc += 1
                    initiator_seg = None
                    init_label = ''
            else: ## curr label is KCHI, but either no initiator or initiator isn't FEM or MAL
                initiator_seg = segment
                init_label = label
    return ctc


def calculate_CTC():
    all_err = []
    cnt = 0
    ctcs = []
    pred_ctcs = []
    mean_abs_error = 0
    avg_ctc = 0
    for file in os.listdir(rttm_dir_name):
        if file[0] == '.': continue
        if not file.endswith('rttm'): continue
        anno_rttm_file = os.path.join(rttm_dir_name, file)
        ctc = gen_ctc_anno(anno_rttm_file)

        ## -- get prediction rttm -- ##
        month_chi = file[:-5]
        if not os.path.exists(os.path.join('/c/Users/Chris-Howard/Desktop/MARVL/validation/audio_val_model_predictions', month_chi, file)):
            continue
        pred_rttm_file = os.path.join('/c/Users/Chris-Howard/Desktop/MARVL/validation/audio_val_model_predictions', month_chi, file)
        pred_ctc = gen_ctc_pred(pred_rttm_file)

        ## calculate pearsons correlation
        ctcs.append(ctc)
        pred_ctcs.append(pred_ctc)

        err = abs(ctc - pred_ctc)
        all_err.append(err)
        mean_abs_error += err
        avg_ctc += ctc
        cnt += 1

    print('mean abs ctc error: ', mean_abs_error / float(cnt))
    print('avg ctc: ', avg_ctc / float(cnt))
    all_err.sort()
    print('all err: ', all_err)

    # calculate Pearson's correlation
    corr, _ = pearsonr(pred_ctcs, ctcs)
    print('Pearsons correlation: %.3f' % corr)


## ------------- run the functions here -------------- ##
# ----------- gen rttms ---------- ##
# for file in os.listdir(trans_dir_name):
#     wc = 0
#     if not file.endswith('txt'): continue
#     transcription_file = open(os.path.join(trans_dir_name, file), 'r', errors='replace')
#     lines = transcription_file.readlines()
#     generate_rttm_file(lines, file)
## -------------------------------------- ##

# calc_avg_word_err()
calculate_DER()
# calculate_CTC()
# calculate_diff_in_DER()


'''
month6_chi_37.txt - fixed
    - s1: child
    - s2: adult female
month6_chi_23.txt -- fixed
    - none
month7_chi_40.txt
    - ?
month13_chi_39.txt
    - ?
month7_chi_08.txt -- fixed
    - adult male
'''
