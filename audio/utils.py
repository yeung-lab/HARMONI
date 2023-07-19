import argparse
import os
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline

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

def gen_ctc(rttm_file, output_dir):
    if not os.path.exists(rttm_file):
        raise Exception("Please enter a valid rttm file!")
    if not os.path.isdir(output_dir):
        raise Exception("Please enter a valid output directory")

    ctc = 0
    initiator_seg = None
    init_label = ''
    ctc_output = os.path.join(output_dir, 'ctc_output.txt')
    f = open(ctc_output, 'w')
    pred_labels = get_indi_annotation(rttm_file)
    for segment, __, label in pred_labels.itertracks(yield_label=True):
        ## need response segment to END after intiator segment --> only count KCHI in CTC
        if 'MAL' in label or 'FEM' in label:
            if initiator_seg != None and init_label == 'KCHI_1':
                if validate_ctc_segs(initiator_seg, segment):
                    toWrite = 'initiator: ' + init_label + str(initiator_seg) + ' -- response: ' + label +  str(segment) + '\n'
                    f.write(toWrite)
                    ctc += 1
                    initiator_seg = None
                    init_label = ''
            else: ## curr label is male or fem, but either no initiator or initiator isn't key child
                initiator_seg = segment
                init_label = label
        elif label == 'KCHI_1':
            if initiator_seg != None and ('MAL' in init_label or 'FEM' in init_label):
                if validate_ctc_segs(initiator_seg, segment):
                    #print('ADDING CTC -- initiator: ', init_label, initiator_seg, ' -- response: ', label, segment)
                    toWrite = 'initiator: ' + init_label + str(initiator_seg) + ' -- response: ' + label +  str(segment) + '\n'
                    f.write(toWrite)
                    ctc += 1
                    initiator_seg = None
                    init_label = ''
            else: ## curr label is KCHI, but either no initiator or initiator isn't FEM or MAL
                initiator_seg = segment
                init_label = label

    last_line = "total ctc: " + str(ctc) + '\n'
    f.write(last_line)
    f.close()
    return ctc
