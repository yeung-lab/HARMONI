'''
- create annotations for both bc and indi rttm files
- for each seg in indi rttm file (have to be this way b/c mapping from individual to most likely class)
    - use seg to crop bc annotation --> new annotation confined by that segment
    - count up total amt of time that individual spker overlaps with each class
- classify each individual speaker to class with most time overlap in bc anno
'''

from pyannote.core import Annotation, Segment
import sys
from collections import defaultdict
import os
import pandas as pd

def get_indi_annotation(file):
    names = ["NA1","uri","NA2","start","duration","NA3","NA4","speaker","NA5","NA6"]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(file, names=names, dtype=dtype, delim_whitespace=True, keep_default_na=False)
    uri = df.at[0, "uri"]
    ## might want to include URI in Annotation()
    indi = Annotation(uri=uri)
    for index, row in df.iterrows():
        start = row['start']
        end = start + row['duration']
        label = row['speaker']
        indi[Segment(start, end)] = label
    return indi

def get_bc_annotation(file):
    names = ["NA1","uri","NA2","start","duration","NA3","NA4","speaker","NA5","NA6"]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(file, names=names, dtype=dtype, delim_whitespace=True, keep_default_na=False)
    uri = df.at[0, "uri"]
    bc = Annotation(uri=uri)
    for index, row in df.iterrows():
        start = row['start']
        end = start + row['duration']
        label = row['speaker']
        if label == 'SPEECH': continue
        bc[Segment(start, end)] = label
    return bc

def gen_mapping(bc_votes):
    spkr_map = {}
    kchi_val = 0
    kchi_lab = -1
    for label in bc_votes.keys():
        votes = bc_votes[label]
        tot_chi = votes['CHI'] + votes['KCHI']
        fem = votes['FEM']
        mal = votes['MAL']
        if max(tot_chi, fem, mal) == tot_chi:
            ## ---- can only have one KCHI -----
            if votes['KCHI'] > votes['CHI'] and votes['KCHI'] > kchi_val:
                spkr_map[label] = 'KCHI'
                if kchi_lab != -1:
                    spkr_map[kchi_lab] = 'CHI'
                kchi_lab = label
                kchi_val = votes['KCHI']
            else:
                spkr_map[label] = 'KCHI'

        elif max(tot_chi, fem, mal) == fem:
            spkr_map[label] = 'FEM'
        else:
            spkr_map[label] = 'MAL'
    return spkr_map

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("usage: python gen_combo_rttm.py <individual.rttm> <bc.rttm> <output.rttm>")
        ## python gen_combo_rttm.py VBx/out_dir_dev/DH_0001.rttm voice_type_classifier/output_voice_type_classifier/DH_0001/all.rttm output.rttm
    indi_rttm = sys.argv[1]
    bc_rttm = sys.argv[2]
    output = sys.argv[3]

    if not os.path.exists(indi_rttm) or not os.path.exists(bc_rttm):
        raise Exception("Please enter valid individual and broadclass rttm files!")

    indi = get_indi_annotation(indi_rttm)
    bc = get_bc_annotation(bc_rttm)
    bc_votes = defaultdict(lambda: defaultdict(float))

    for seg in indi.itersegments():
        indi_label = indi[seg]
        rel_bc = bc.crop(seg)
        bc_labels = rel_bc.labels()
        for label in bc_labels:
            bc_votes[indi_label][label] += rel_bc.label_duration(label)
    spkr_mapping = gen_mapping(bc_votes)
    #print('speaker mapping: ', spkr_mapping)
    spk_num = {'KCHI': 1, 'FEM': 1, 'CHI': 1, 'MAL': 1}
    final_labels = {}
    for label in spkr_mapping.keys():
        class_type = spkr_mapping[label]
        num = spk_num[class_type]
        new_label = class_type + '_' + str(num)
        final_labels[label] = new_label
        spk_num[class_type] += 1
    #print('final labels: ', final_labels)
    combo_anno = indi.rename_labels(final_labels)

    with open(output, 'w') as file:
        combo_anno.write_rttm(file)
