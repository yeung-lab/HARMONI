Audio component of HARMONI optimized for annotating the SEEDLings dataset

## Additional installation:
Please rebuild the x-vector extractor file used for individual speaker diarization.
```
cd VBx
zip -s 0 split_xvector_extractor.txt.zip --out unsplit_xvector_extractor.txt.zip ## unsplit the file
unzip unsplit_xvector_extractor.txt.zip ## unzip the file
cd -
mkdir VBx/inference_files
```

## Usage:
Example command to run the audio model on a video file.
```
python run.py path_to_mp4_file output_folder
```

## Output:
The results will be stored in the `./output`` folder where you will find four primary files:
- ALICE_output.txt --> prediction of the number of words, syllables, and phonemes spoken by adults in the audio clip
- ALICE_output_utterances.txt --> predictions of words, syllables, and phonemes for each of the different identified segments of speaking in the clip
- ctc_output.txt --> the occurances of conversation turns between adult and child in the clip
- audio_file_name.rttm --> the individual speaker diarization file with broad class classifications
