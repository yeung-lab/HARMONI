# Things to check before applying

Make sure your recordings are under the **.wav** format and have been sampled at **16 kHz**, to do so you can use :

```bash
soxi /path/to/my_recording.wav
```

You also need to make sure that you've activated the conda environment you created : 

```bash
conda activate pyannote # or source activate pyannote, depending on your config
```

Last but not least, make sure that you are in the ***voice_type_classifier*** repository.

# Applying

The code consists mainly of a wrapper of pyannote-audio's code responsible for applying a model.
If you want to go further by retraining the model, you should go around [here](https://github.com/MarvinLvn/pyannote-audio/tree/voice_type_classifier).

You can apply our pre-trained model by typing :

```bash
./apply.sh /path/to/my_recordings/ --device=gpu
```

where **/path/to/my_recordings/** is a folder containing audio recordings in the .wav format (single-channel, 16kHz). 
This will actually apply 5 models (each for one of the class), which can take some time.
The flag ***--device=gpu*** indicates if the model should be run on gpu or not. If this flag is not provided, the model will run on CPU.

You can also apply the model to a single audio file :

```bash
./apply.sh /path/to/my_recordings/my_recording.wav --device=gpu
```
# Output format

The above commands will generate a ***output_voice_type_classifier*** folder, itself containing sub-folders named after the folder you applied the model on. 
Let's assume we applied the model on the ***my_recordings1***  and ***my_recordings2*** folder :

```bash
ls output_voice_type_classifier/
```

returns

```bash
my_recordings1  my_recordings2
```

Now, let's have a look at what ***my_recordings1*** contains :

```bash
ls output_voice_type_classifier/my_recordings1
```

returns 

```bash
CHI.rttm  FEM.rttm  KCHI.rttm  MAL.rttm  SPEECH.rttm  all.rttm
```

Each of the rttm files contains lines that look like :

```bash
SPEAKER	my_recording_number1	1	33.429	0.513	<NA>	<NA>	KCHI	<NA>	<NA>
SPEAKER	my_recording_number1	1	35.785	1.117	<NA>	<NA>	KCHI	<NA>	<NA>
SPEAKER	my_recording_number2	1	0.107	0.154	<NA>	<NA>	KCHI	<NA>	<NA>
SPEAKER	my_recording_number2	1	1.284	0.332	<NA>	<NA>	KCHI	<NA>	<NA>
```

Where :
- each line corresponds to a speech/vocalization utterance
- the second column is the name of the file concerned by this utterance
- the fourth column is the onset of the utterance
- the fifth column is the duration of the utterance
- the eigth column is the voice type, amongst [KCHI, CHI, MAL, FEM, SPEECH]

***CHI.rttm*** contains only utterances produced by other children (according to the model).

***MAL.rttm*** contains only utterances produced by male speakers (according to the model)

And so on ...

***all.rttm*** is the concatenation (sorted by filename and onset) of ***{KCHI,CHI,MAL,FEM,SPEECH}.rttm***