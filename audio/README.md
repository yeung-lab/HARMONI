# seedlings_audio
Audio component of AI annotation tool optimized for annotating the SEEDLings dataset


## Requirements:
This tool has been developed and tested fo Linux and macOS environments. Windows users may encounter problems with some package versions.

Packages:

- Conda (https://docs.conda.io/en/latest/)
- CMake (pip install cmake or conda install cmake)
(other packages automatically installed by conda environment)


## Installation:
First, go into the VBx subdirectory and rebuild the x-vector extractor file used for individual speaker diarization
```
zip -s 0 split_xvector_extractor.txt.zip --out unsplit_xvector_extractor.txt.zip ## unsplit the file
unzip unsplit_xvector_extractor.txt.zip ## unzip the file
```

Then, make sure you have Conda and Cmake installed
- Create the conda environment installing all the dependencies:

```
cd seedlings_audio
conda env create -f process_audio.yml
conda activate process_audio
```

## Usage:
This tool only accepts audio in the following format:
```
- codec: PCM S16 LE
- channel: mono
- sample rate: 16khz
- bits per sample: 16
- wrapper: wav
```
To convert your audio file into this format, use the preprocessing.py script
```
python preprocess.py <audio_file_here>
```
Once you have a correctly formatted audio file, run the tool using the process_audio.py script. Only use the GPU flag if you have a GPU.
```
python process_audio.py correctly_formated_audio.wav <--gpu>
```
*Note: this script currently only takes a single file

## Output:
The results from the process_audio.py script will be stored in the output folder where you will find four primary files:
- ALICE_output.txt --> prediction of the number of words, syllables, and phonemes spoken by adults in the audio clip
- ALICE_output_utterances.txt --> predictions of words, syllables, and phonemes for each of the different identified segments of speaking in the clip
- ctc_output.txt --> the occurances of conversation turns between adult and child in the clip
- audio_file_name.rttm --> the individual speaker diarization file with broad class classifications
