#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Marvin LAVECHIN - marvinlavechin@gmail.com

"""Multilabel detection detection pipeline"""

from pathlib import Path
from typing import Iterator
from typing import Text
from typing import Union

import numpy as np
from pyannote.audio.features.wrapper import Wrapper as FeatureExtractionWrapper
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature
from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.database import get_annotated

class MultilabelDetection(Pipeline):
    """Multilabel detection pipeline

    Parameters
    ----------
    scores : Text or Path, optional
        Describes how raw multilabel detection scores should be obtained.
        It can be either the name of a torch.hub model, or the path to the
        output of the validation step of a model trained locally, or the path
        to scores precomputed on disk. Defaults to "@labels_scores" that indicates
        that protocol files provide the scores in the "labels_scores" key.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.


    Hyper-parameters
    ----------------
    onset, offset : `float`
        Onset/offset detection thresholds
    min_duration_on, min_duration_off : `float`
        Minimum duration in either state (speech or not)
    pad_onset, pad_offset : `float`
        Padding duration.
    """

    def __init__(self, label_list, considered_label, scores: Union[Text, Path] = None,
                       fscore: bool = False, precision=None):
        super().__init__()

        if scores is None:
            scores = "@labels_scores"

        self.label_list = label_list
        self.considered_label = considered_label
        self.scores = scores
        self._scores = FeatureExtractionWrapper(self.scores)

        self.fscore = fscore
        self.precision = precision


        # hyper-parameters
        self.onset = Uniform(0., 1.)
        self.offset = Uniform(0., 1.)
        self.min_duration_on = Uniform(0., 2.)
        self.min_duration_off = Uniform(0., 2.)
        self.pad_onset = Uniform(-1., 1.)
        self.pad_offset = Uniform(-1., 1.)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            pad_onset=self.pad_onset,
            pad_offset=self.pad_offset)

    def __call__(self, current_file: dict) -> Annotation:
        """Apply multilabel detection

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol. May contain a
            'labels_scores' key providing precomputed scores.

        Returns
        -------
        speech : `pyannote.core.Annotation`
            Speech regions.
        """

        # This should be moved to pyannote-core/pyannote/core/utils/generators.py
        def constant_generator(elem) -> Iterator[str]:
            while True:
                yield elem

        labels_scores = self._scores(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(labels_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(labels_scores.data) if self.log_scale_ \
               else labels_scores.data

        col_index = self.label_list.index(self.considered_label)
        activation_prob = SlidingWindowFeature(data[:, col_index], labels_scores.sliding_window)
        activation = self._binarize.apply(activation_prob)

        activation.uri = current_file['uri']

        return activation.to_annotation(generator=constant_generator(self.considered_label),
                                        modality=self.considered_label)

    def get_metric(self, parallel=False) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0,
                                                    skip_overlap=False,
                                                    parallel=parallel)
        else:
            return DetectionErrorRate(collar=0.0,
                                      skip_overlap=False,
                                      parallel=parallel)

    def loss(self, current_file: dict, hypothesis=None):
        reference = current_file['annotation']
        uem = get_annotated(current_file)

        precision = DetectionPrecision()
        recall = DetectionRecall()

        p = precision(reference, hypothesis, uem=uem)
        r = recall(reference, hypothesis, uem=uem)

        if p > self.precision:
            return 1. - r
        else:
            return 1. + (1. - p)