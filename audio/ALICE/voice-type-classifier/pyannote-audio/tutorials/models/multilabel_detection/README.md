> The MIT License (MIT)
>
> Copyright (c) 2017-2020 CNRS
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
> AUTHOR
> Marvin LAVECHIN - marvinlavechin@gmail.com

# End-to-end multilabel detection with `pyannote.audio`

This tutorial will teach you train a multilabel detection model.
It is highly similar to the speech activity detection tutorial.

## Table of contents
- [Citation](#citation)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for multilabel detection, please cite the following papers:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

```bibtex
@inproceedings{Lavechin2020,
  Title = {{End-to-end Domain-Adversarial Voice Activity Detection}},
  Author = {{Lavechin}, Marvin and {Gill}, Marie-Philippe and {Bousbib}, Ruben and {Bredin}, Herv{\'e} and {Garcia-Perera}, Leibny Paola},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```


## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-audio` relies on a configuration file defining the experimental setup:

```bash
$ export EXP_DIR=tutorials/models/multilabel_detection
$ cat ${EXP_DIR}/config.yml
```
```yaml
# (Optional) Here we use a LabelMapper as a preprocessor
# for the input annotations. This LabelMapper,
# will map the input labels to their targets.
# Hence, every utterances of Bob will be transformed
# as utterances belonging to the MAL class.
preprocessors:
  annotation:
    name: pyannote.database.util.LabelMapper
    params:
      keep_missing: False
      mapping:
        "Bob": "MAL"
        "Alice": "FEM"
        "Nolan": "CHI"
        "Arthur": "MAL"
        ...

# A multilabel detection model is trained.
# Here, training relies on 2s-long audio chunks,
# batches of 64 audio chunks, and saves model to
# disk every one (1) day worth of audio.
# The labels_spec specifies how to build the classes
# of the model given the raw label.

# a) Here, we consider 4 regular labels which are
# KCHI, CHI, MAL, FEM. The latter will account for
# 4 dimensions in the predicted vector. This type
# of labels is activated whenever a utterance with
# the label of interest is encountered in the input
# annotation.

# b) 1 union label called "SPEECH" which is made of the union
# of the classes KCHI, CHI, FEM, MAL, UNK. Hence, the SPEECH
# class will be activated whenever one (or more) of the
# listed class is activated. Note that UNK contributes to
# activate the SPEECH class but is not predicted by the model

# c) 1 intersection label called "OVL" which is made of the
# intersection of the classes KCHI, CHI, FEM, MAL, UNK. Hence,
# the OVL class will be activated whenever at least 2 of the listed
# classes are activated at the same time.

# To sum up, our model will output a vector of 6 scores :
# [KCHI, CHI, MAL, FEM, SPEECH, UNK]
task:
   name: MultilabelDetection
   params:
      duration: 2.0
      batch_size: 64
      per_epoch: 1
      labels_spec:
        regular: ['KCHI', 'CHI', 'MAL', 'FEM']
        union:
          SPEECH: ['KCHI', 'CHI', 'FEM', 'MAL', 'UNK']
        intersection:
          OVL: ['KCHI', 'CHI', 'FEM', 'MAL', 'UNK']

# Data augmentation is applied during training.
# Here, it consists in additive noise from the
# MUSAN database, with random signal-to-noise
# ratio between 5 and 20 dB
data_augmentation:
   name: AddNoise
   params:
      snr_min: 5
      snr_max: 20
      collection: MUSAN.Collection.BackgroundNoise

# Since we are training an end-to-end model, the
# feature extraction step simply returns the raw
# waveform.
feature_extraction:
   name: RawAudio
   params:
      sample_rate: 16000

# We use the PyanNet architecture in Figure 2 of
# pyannote.audio introductory paper. More details
# about the architecture and its parameters can be
# found directly in PyanNet docstring.
architecture:
   name: pyannote.audio.models.PyanNet
   params:
      rnn:
         unit: LSTM
         hidden_size: 128
         num_layers: 2
         bidirectional: True
      ff:
         hidden_size: [128, 128]

# We use a constant learning rate of 1e-2
scheduler:
   name: ConstantScheduler
   params:
      learning_rate: 0.01
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training subset of AMI database for 200 epochs:

```bash
$ pyannote-audio mlt train --subset=train --to=200 --parallel=4 ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `TRN_DIR` (defined below). One can also follow along the training process using [tensorboard](https://github.com/tensorflow/tensorboard):
```bash
$ tensorboard --logdir=${EXP_DIR}
```

![tensorboard screenshot](tb_train.png)


## Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing on the development set, one can use the `validate` mode.

```bash
$ export TRN_DIR=${EXP_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train
$ pyannote-audio mlt validate --subset=development --from=10 --to=200 --every=10 ${TRN_DIR} AMI.SpeakerDiarization.MixHeadset
```
It can be run while the model is still training and evaluates the model every 10 epochs. This will create a bunch of files in `VAL_DIR` (defined below). 

In practice, it tunes a simple speech activity detection pipeline for each of the class of interest every 10 epochs and stores the best hyper-parameter configuration on disk (i.e. the one that maximizes the average detection f-score across classes):

```bash
$ export VAL_DIR = ${TRN_DIR}/validate_average_detection_fscore/AMI.SpeakerDiarization.MixHeadset.development
$ cat ${VAL_DIR}/params.yml
```
```yaml
CHI:
  detection_fscore: 0.22627409456545222
  params:
    min_duration_off: 0.1
    min_duration_on: 0.1
    offset: 0.24
    onset: 0.24
    pad_offset: 0.0
    pad_onset: 0.0
FEM:
  detection_fscore: 1.0
  params:
    min_duration_off: 0.1
    min_duration_on: 0.1
    offset: 0.45
    onset: 0.45
    pad_offset: 0.0
    pad_onset: 0.0
KCHI:
  detection_fscore: 1.0
  params:
    min_duration_off: 0.1
    min_duration_on: 0.1
    offset: 0.62
    onset: 0.622
    pad_offset: 0.0
    pad_onset: 0.0
MAL:
  detection_fscore: 0.1183285794101212
  params:
    min_duration_off: 0.1
    min_duration_on: 0.1
    offset: 0.33
    onset: 0.33
    pad_offset: 0.0
    pad_onset: 0.0
SPEECH:
  detection_fscore: 0.3199697547297614
  params:
    min_duration_off: 0.1
    min_duration_on: 0.1
    offset: 0.56
    onset: 0.56
    pad_offset: 0.0
    pad_onset: 0.0
average_detection_fscore: 0.5329144857410669
epoch: 10
labels_spec:
  intersection: {}
  regular:
  - KCHI
  - CHI
  - MAL
  - FEM
  union:
    SPEECH:
    - KCHI
    - CHI
    - MAL
    - FEM
    - SPEECH
```

See `pyannote.audio.pipeline.multilabel_detection.MultilabelDetection` for details on the role of each parameter.

Like for training, one can also use [tensorboard](https://github.com/tensorflow/tensorboard) to follow the validation process:

```bash
$ tensorboard --logdir=${EXP_DIR}
```

![tensorboard screenshot](tb_validate.png)


## Application
([↑up to table of contents](#table-of-contents))

Now that we know how the model is doing, we can apply it on test files of the AMI database: 

```bash
$ pyannote-audio mlt apply --subset=test ${VAL_DIR} AMI.SpeakerDiarization.MixHeadset 
```

Raw model output and speech activity detection results, for each of the class, will be dumped into the following directory: `${VAL_DIR}/apply/{BEST_EPOCH}`.

## More options

For more options, see:

```bash
$ pyannote-audio --help
```

That's all folks!