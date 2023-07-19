import os
import tqdm
import pandas as pd
from pyannote.core import Segment, Timeline, Annotation

'''
Apply VTC to eval files
'''
def run_vtc_inference_over_val(audio_dir):
    for file in tqdm.tqdm(os.listdir(audio_dir)):
        if not file.endswith('.wav'): continue
        if file.startswith('.'): continue
        dirname = file[:-4]
        if os.path.exists(os.path.join('output_voice_type_classifier', dirname)):
            print('Skipping file {} ...'.format(file))
            continue
        print('Evaluating file {} ...'.format(file))
        os.system("voice_type_classifier/apply.sh {0}".format(os.path.join(audio_dir, file)))

def run_pyannote_dev_inference_over_val(audio_dir):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

    output_dir = 'pyannote_dev_val_rttms/'
    for audio_file in os.listdir(audio_dir):
        if audio_file.startswith('.'): continue
        if not audio_file.endswith('.wav'): continue
        print('audio file: ', audio_file)
        file = os.path.join(audio_dir, audio_file)
        speech_activity_detection = pipeline({'audio': file})

        # dump result to disk using RTTM format
        with open(os.path.join(output_dir, audio_file[:-4] + '.rttm'), 'w') as f:
            speech_activity_detection.write_rttm(f)


def get_df_from_rttm(file):
    names = ["NA1", "uri", "NA2", "start", "duration", "NA3", "NA4", "speaker", "NA5", "NA6"]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    df = pd.read_csv(
            file,
            names=names,
            dtype=dtype,
            delim_whitespace=True,
            keep_default_na=False,
        )
    return df

'''
Create pyannote annotation object from given rttm file
'''
def get_rttm_annotation(file):
    df = get_df_from_rttm(file)
    anno = Annotation()
    for index, row in df.iterrows():
        start = row['start']
        end = start + row['duration']
        label = row['speaker']
        anno[Segment(start, end)] = label
    return anno

def get_pred_timeline(file):
    df = get_df_from_rttm(file)
    anno = Timeline()
    for index, row in df.iterrows():
        start = row['start']
        end = start + row['duration']
        label = row['speaker']
        anno = anno.add(Segment(start, end))
    return anno


'''
Clip transcriptions according to the predicted VTC sad
'''
def clip_transcriptions(trans_dir, sad_dir, output_dir):
    for rttm_trans in os.listdir(trans_dir):
        if not rttm_trans.endswith('.rttm'): continue
        print('Analyzing rttm {}'.format(rttm_trans))
        trans_anno = get_rttm_annotation(os.path.join(trans_dir, rttm_trans))

        pred_sad_file = os.path.join(sad_dir, rttm_trans[:-5], 'SPEECH.rttm')
        if not os.path.exists(pred_sad_file):
            print("Error: could not find SAD file for {}".format(rttm_trans[:-5]))
        pred_sad_support = get_pred_timeline(pred_sad_file)
        # print('transcript annotation: ', trans_anno)
        # print('pred support: ', pred_sad_support)
        cropped_trans_anno = trans_anno.crop(pred_sad_support, mode="intersection")
        # print('transcript annotation: ', trans_anno)
        # print('cropped trans: ', cropped_trans_anno)
        output_rttm = os.path.join(output_dir, rttm_trans)
        with open(output_rttm, 'w') as file:
            cropped_trans_anno.write_rttm(file)

'''
Clip transcriptions according to the predicted VTC sad leaving missed detections!
'''
def clip_transcriptions_w_md(trans_dir, sad_dir, output_dir):
    for rttm_trans in os.listdir(trans_dir):
        if not rttm_trans.endswith('.rttm'): continue
        print('Analyzing rttm {}'.format(rttm_trans))
        trans_anno_1 = get_rttm_annotation(os.path.join(trans_dir, rttm_trans))
        # print('trans anno: ', trans_anno_1)
        # print('transcript annotation: ', trans_anno_1)
        pred_sad_file = os.path.join(sad_dir, rttm_trans[:-5], 'SPEECH.rttm')
        if not os.path.exists(pred_sad_file):
            print("Error: could not find SAD file for {}".format(rttm_trans[:-5]))
        pred_sad_support = get_pred_timeline(pred_sad_file)
        pred_sad_inv = pred_sad_support.gaps()
        if len(pred_sad_support) > 0:
            start = 0.0
            end = pred_sad_support[0].start
            pred_sad_inv.add(Segment(start, end))

            start = pred_sad_support[-1].end
            end = 60.0
            pred_sad_inv.add(Segment(start, end))
        # print('sad support: ', pred_sad_support)
        cropped_trans_anno_1 = trans_anno_1.crop(pred_sad_inv, mode="strict")
        # print('missed detection components: ', cropped_trans_anno_1)

        trans_anno_2 = get_rttm_annotation(os.path.join(trans_dir, rttm_trans))
        cropped_trans_anno_2 = trans_anno_2.crop(pred_sad_support, mode="intersection")
        final_trans_anno = cropped_trans_anno_1.update(cropped_trans_anno_2)
        # print('cropped overlap segments: ', cropped_trans_anno_2)
        # print('final: ', final_trans_anno)
        # print('pred support: ', pred_sad_support)
        # print('final cropped trans: ', final_trans_anno)

        output_rttm = os.path.join(output_dir, rttm_trans)
        with open(output_rttm, 'w') as file:
            final_trans_anno.write_rttm(file)

'''
Clip transcriptions according to the predicted pyannote dev sad leaving missed detections!
'''
def clip_transcriptions_w_md_pyan_dev(trans_dir, sad_dir, output_dir):
    for rttm_trans in os.listdir(trans_dir):
        if not rttm_trans.endswith('.rttm'): continue
        print('Analyzing rttm {}'.format(rttm_trans))
        trans_anno_1 = get_rttm_annotation(os.path.join(trans_dir, rttm_trans))
        # print('trans anno: ', trans_anno_1)
        # print('transcript annotation: ', trans_anno_1)
        pred_sad_file = os.path.join(sad_dir, rttm_trans)
        if not os.path.exists(pred_sad_file):
            print("Error: could not find SAD file for {}".format(rttm_trans[:-5]))
        pred_sad_support = get_pred_timeline(pred_sad_file)
        pred_sad_inv = pred_sad_support.gaps()
        if len(pred_sad_support) > 0:
            start = 0.0
            end = pred_sad_support[0].start
            pred_sad_inv.add(Segment(start, end))

            start = pred_sad_support[-1].end
            end = 60.0
            pred_sad_inv.add(Segment(start, end))
        # print('sad support: ', pred_sad_support)
        cropped_trans_anno_1 = trans_anno_1.crop(pred_sad_inv, mode="strict")
        # print('missed detection components: ', cropped_trans_anno_1)

        trans_anno_2 = get_rttm_annotation(os.path.join(trans_dir, rttm_trans))
        cropped_trans_anno_2 = trans_anno_2.crop(pred_sad_support, mode="intersection")
        final_trans_anno = cropped_trans_anno_1.update(cropped_trans_anno_2)
        # print('cropped overlap segments: ', cropped_trans_anno_2)
        # print('final: ', final_trans_anno)
        # print('pred support: ', pred_sad_support)
        # print('final cropped trans: ', final_trans_anno)

        output_rttm = os.path.join(output_dir, rttm_trans)
        with open(output_rttm, 'w') as file:
            final_trans_anno.write_rttm(file)


## first, run VTC over all the eval files
# run_vtc_inference_over_val('/Volumes/data_hauler/eval_audio/')
# run_pyannote_dev_inference_over_val('/Volumes/data_hauler/eval_audio/')
trans_dir = '/Volumes/data_hauler/MARVL_PC_NEW/validation/seedlings_audio_transcriptions/rttms/'
sad_dir = 'voice_type_classifier/output_voice_type_classifier/'
output_dir = 'clipped_seedlings_eval_transcripts/'
output_dir_wmd = 'clipped_seedlings_eval_transcripts_wmd_correct'
sad_dir_pyan = 'pyannote_dev_val_rttms'
output_dir_pyan = 'clipped_seedlings_eval_transcripts_wmd_pyan_dev'
# clip_transcriptions(trans_dir, sad_dir, output_dir)
# clip_transcriptions_w_md(trans_dir, sad_dir, output_dir_wmd)
clip_transcriptions_w_md_pyan_dev(trans_dir, sad_dir_pyan, output_dir_pyan)
