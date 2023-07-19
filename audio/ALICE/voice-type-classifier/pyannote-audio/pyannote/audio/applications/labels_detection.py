#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

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


import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.optimize
import torch
import yaml
from pyannote.audio.applications.base import create_zip
from pyannote.audio.features import Precomputed
from pyannote.audio.features import Pretrained
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.wrapper import Wrapper
from pyannote.audio.labeling.tasks import MultilabelDetection as MultilabelTask
from pyannote.audio.pipeline import MultilabelDetection \
    as MultilabelDetectionPipeline
from pyannote.database import FileFinder
from pyannote.database import get_annotated
from pyannote.database import get_protocol
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision
from sortedcontainers import SortedDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Literal

from .base_labeling import BaseLabeling


class MultilabelDetection(BaseLabeling):

    Pipeline = MultilabelDetectionPipeline

    def validate_helper_func(self, current_file, pipeline=None, metric=None, precision=None, recall=None, label=None):

        # Get to know how to derive the considered label
        if label in self.task_.labels_spec["regular"]:
            derivation_type = "regular"
        elif label in self.task_.labels_spec["union"]:
            derivation_type = "union"
        elif label in self.task_.labels_spec["intersection"]:
            derivation_type = "intersection"
        else:
            raise ValueError("%s not found in training labels : %s"
                             % (label, self.task_.label_names))

        if derivation_type == "regular":
            reference = current_file["annotation"].subset([label])
        else:
            reference = MultilabelTask.derives_label(current_file["annotation"],
                                                     derivation_type=derivation_type,
                                                     meta_label=label,
                                                     regular_labels=self.task_.labels_spec[derivation_type][label])

        uem = get_annotated(current_file)
        hypothesis = pipeline(current_file)
        if precision is not None:
            p = precision(reference, hypothesis, uem=uem)
            r = recall(reference, hypothesis, uem=uem)
            return p, r
        else:
            return metric(reference, hypothesis, uem=uem)

    def validation_criterion(self, protocol, precision=None, **kwargs):
        if precision:
            return f'recall@{100 * precision:.2f}precision'
        else:
            return f'average_detection_fscore'

    def validate_epoch(self,
                       epoch,
                       validation_data,
                       device=None,
                       batch_size=32,
                       n_jobs=1,
                       duration=None,
                       step=0.25,
                       precision=None,
                       **kwargs):

        target_precision = precision
        label_names = self.task_.label_names

        # compute (and store) SAD scores
        pretrained = Pretrained(validate_dir=self.validate_dir_,
                                epoch=epoch,
                                duration=duration,
                                step=step,
                                batch_size=batch_size,
                                device=device)

        for current_file in validation_data:
            # Get N scores per frame such as returned by the pretrained model
            current_file['scores'] = pretrained(current_file)

        if target_precision:
            result = {'metric': self.validation_criterion(None, precision=precision),
                      'minimize': False,
                      'labels': label_names}
            aggregated_metric = 0

            for considered_label in label_names:
                lower_alpha = 0.
                upper_alpha = 1.
                best_alpha = .5 * (lower_alpha + upper_alpha)
                best_recall = 0.
                pipeline = MultilabelDetectionPipeline(scores="@scores",
                                         label_list=label_names,
                                         considered_label=considered_label,
                                         precision=target_precision)

                # dichotomic search to find threshold that maximizes recall
                # while having at least `target_precision`

                for _ in range(10):

                    current_alpha = .5 * (lower_alpha + upper_alpha)
                    pipeline.instantiate({'onset':  current_alpha,
                                          'offset':  current_alpha,
                                          'min_duration_on': 0.100,
                                          'min_duration_off': 0.100,
                                          'pad_onset':  0.,
                                          'pad_offset':  0.})

                    _precision = DetectionPrecision(parallel=True)
                    _recall = DetectionRecall(parallel=True)

                    validate = partial(self.validate_helper_func,
                                       pipeline=pipeline,
                                       precision=_precision,
                                       recall=_recall,
                                       label=considered_label)
                    if n_jobs > 1:
                        _ = self.pool_.map(validate, validation_data)
                    else:
                        for file in validation_data:
                            _ = validate(file)

                    _precision = abs(_precision)
                    _recall = abs(_recall)

                    if _precision < target_precision:
                        # precision is not high enough:  try higher thresholds
                        lower_alpha = current_alpha
                    else:
                        upper_alpha = current_alpha
                        if _recall > best_recall:
                            best_recall = _recall
                            best_alpha = current_alpha

                result[considered_label] = {}
                result[considered_label]['pipeline'] = pipeline.instantiate({'onset': best_alpha,
                                                                             'offset': best_alpha,
                                                                             'min_duration_on': 0.100,
                                                                             'min_duration_off': 0.100,
                                                                             'pad_onset': 0.,
                                                                             'pad_offset': 0.})
                result[considered_label]['minimize'] = False
                result[considered_label]['metric'] = f'recall@{100 * target_precision:.2f}precision'
                result[considered_label]['value'] = best_recall
                aggregated_metric += result[considered_label]['value']

            aggregated_metric /= len(label_names)
            result['value'] = aggregated_metric

            return result
        else:
            # Detection Error Rate or Fscore validation
            def fun(threshold, considered_label):
                pipeline.instantiate({'onset': threshold,
                                      'offset': threshold,
                                      'min_duration_on': 0.100,
                                      'min_duration_off': 0.100,
                                      'pad_onset': 0.,
                                      'pad_offset': 0.})
                metric = pipeline.get_metric(parallel=True)
                validate = partial(self.validate_helper_func,
                                   pipeline=pipeline,
                                   metric=metric,
                                   label=considered_label,
                                   precision=precision)
                if n_jobs > 1:
                    _ = self.pool_.map(validate, validation_data)
                else:
                    for file in validation_data:
                        _ = validate(file)

                return 1. - abs(metric)

            result = {'metric': self.validation_criterion(None),
                      'minimize': False,
                      'labels': label_names}

            aggregated_metric = 0
            for considered_label in label_names:
                pipeline = self.Pipeline(scores="@scores",
                                         fscore=True,
                                         label_list=label_names,
                                         considered_label=considered_label)

                res = scipy.optimize.minimize_scalar(
                    fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10}, args=considered_label)

                threshold = res.x.item()

                result[considered_label] = {}
                result[considered_label]['pipeline'] = pipeline.instantiate({'onset': threshold,
                                                                             'offset': threshold,
                                                                             'min_duration_on': 0.100,
                                                                             'min_duration_off': 0.100,
                                                                             'pad_onset': 0.,
                                                                             'pad_offset': 0.})
                result[considered_label]['metric'] = 'detection_fscore'
                result[considered_label]['value'] = float(1.-res.fun)
                aggregated_metric += result[considered_label]['value']

            aggregated_metric /= len(label_names)
            result['value'] = aggregated_metric

            return result

    def validate(self, protocol: str,
                       subset: str = 'development',
                       every: int = 1,
                       start: Union[int, Literal['last']] = 1,
                       end: Union[int, Literal['last']] = 100,
                       chronological: bool = False,
                       device: Optional[torch.device] = None,
                       batch_size: int = 32,
                       n_jobs: int = 1,
                       precision: int = None,
                       **kwargs):
        # use last available epoch as starting point
        if start == 'last':
            start = self.get_number_of_epochs() - 1

        # use last available epoch as end point
        if end == 'last':
            end = self.get_number_of_epochs() - 1

        criterion = self.validation_criterion(protocol, precision,
                                              **kwargs)

        validate_dir = Path(self.VALIDATE_DIR.format(
            train_dir=self.train_dir_,
            _criterion=f'_{criterion}' if criterion is not None else '',
            protocol=protocol, subset=subset))

        params_yml = validate_dir / 'params.yml'

        validate_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(validate_dir),
                               purge_step=start)

        self.validate_dir_ = validate_dir

        validation_data = self.validate_init(protocol, subset=subset)

        if n_jobs > 1:
            self.pool_ = multiprocessing.Pool(n_jobs)

        progress_bar = tqdm(unit='iteration')

        for i, epoch in enumerate(
            self.validate_iter(start=start, end=end, step=every,
                               chronological=chronological)):

            # {'metric': 'detection_error_rate',
            #  'minimize': True,
            #  'value': 0.9,
            #  'pipeline': ...}
            details = self.validate_epoch(epoch,
                                          validation_data,
                                          protocol=protocol,
                                          subset=subset,
                                          device=device,
                                          batch_size=batch_size,
                                          n_jobs=n_jobs,
                                          precision=precision,
                                          **kwargs)

            # initialize
            if i == 0:
                # what is the name of the metric?
                metric = details['metric']
                # should the metric be minimized?
                minimize = details['minimize']

                # epoch -> value dictionary
                values = SortedDict()

                # load best epoch and value from past executions
                if params_yml.exists():
                    with open(params_yml, 'r') as fp:
                        params = yaml.load(fp, Loader=yaml.SafeLoader)
                    best_epoch = params['epoch']
                    best_value = params[metric]
                    values[best_epoch] = best_value

            # metric value for current epoch
            values[epoch] = details['value']

            # send value to tensorboard
            writer.add_scalar(
                f'validate/{protocol}.{subset}/{metric}',
                values[epoch], global_step=epoch)

            # keep track of best value so far
            if minimize:
                best_epoch = values.iloc[np.argmin(values.values())]
                best_value = values[best_epoch]

            else:
                best_epoch = values.iloc[np.argmax(values.values())]
                best_value = values[best_epoch]

            # if current epoch leads to the best metric so far
            # store both epoch number and best pipeline parameter to disk
            if best_epoch == epoch:

                best = {
                    metric: best_value,
                    'epoch': epoch,
                    'labels_spec': self.task_.labels_spec
                }
                for label in self.task_.label_names:
                    if label in details:
                        best[label] = {}
                        pipeline = details[label]['pipeline']
                        best[label]['params'] = pipeline.parameters(instantiated=True)
                        submetric = details[label]['metric']
                        best[label][submetric] = details[label]['value']

                with open(params_yml, mode='w') as fp:
                    fp.write(yaml.dump(best, default_flow_style=False))

                # create/update zip file for later upload to torch.hub
                hub_zip = create_zip(validate_dir)

            # progress bar
            desc = (f'{metric} | '
                    f'Epoch #{best_epoch} = {100 * best_value:g}% (best) | '
                    f'Epoch #{epoch} = {100 * details["value"]:g}%')
            progress_bar.set_description(desc=desc)
            progress_bar.update(1)

    # TODO: add support for torch.hub models directly in docopt
    @staticmethod
    def apply_pretrained(validate_dir: Path,
                         protocol_name: str,
                         subset: Optional[str] = "test",
                         duration: Optional[float] = None,
                         step: float = 0.25,
                         device: Optional[torch.device] = None,
                         batch_size: int = 32,
                         pretrained: Optional[str] = None,
                         Pipeline: type = None,
                         **kwargs):
        """Apply pre-trained model

        Parameters
        ----------
        validate_dir : Path
        protocol_name : `str`
        subset : 'train' | 'development' | 'test', optional
            Defaults to 'test'.
        duration : `float`, optional
        step : `float`, optional
        device : `torch.device`, optional
        batch_size : `int`, optional
        pretrained : `str`, optional
        Pipeline : `type`
        """

        if pretrained is None:
            pretrained = Pretrained(validate_dir=validate_dir,
                                    duration=duration,
                                    step=step,
                                    batch_size=batch_size,
                                    device=device)
            output_dir = validate_dir / 'apply' / f'{pretrained.epoch_:04d}'
        else:

            if pretrained in torch.hub.list('pyannote/pyannote-audio'):
                output_dir = validate_dir / pretrained
            else:
                output_dir = validate_dir

            pretrained = Wrapper(pretrained,
                                 duration=duration,
                                 step=step,
                                 batch_size=batch_size,
                                 device=device)
        params = {}

        try:
            params['classes'] = pretrained.classes
        except AttributeError as e:
            pass
        try:
            params['dimension'] = pretrained.dimension
        except AttributeError as e:
            pass

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=pretrained.sliding_window,
            **params)

        # file generator
        preprocessors = getattr(pretrained, "preprocessors_", dict())
        if "audio" not in preprocessors:
            preprocessors["audio"] = FileFinder()
        if 'duration' not in preprocessors:
            preprocessors['duration'] = get_audio_duration
        protocol = get_protocol(protocol_name,
                                progress=True,
                                preprocessors=preprocessors)

        for current_file in getattr(protocol, subset)():
            fX = pretrained(current_file)
            precomputed.dump(current_file, fX)

        # do not proceed with the full pipeline
        # when there is no such thing for current task
        if Pipeline is None:
            return

        # Dirty hack to check if the validation optimized fscore or detection error rate
        # In which case, we'll use the same metric
        fscore = 'detection_fscore' in str(pretrained.validate_dir).split('/')[-2]

        for label in precomputed.classes_:

            # Initialization of the pipeline associated to this label
            pipeline = Pipeline(scores=output_dir,
                                label_list=precomputed.classes_,
                                considered_label=label,
                                fscore=fscore)

            pipeline.instantiate(getattr(pretrained, "{}_pipeline_params_".format(label)))

            # Decides which type of label we're currently handling
            # so that we know how to derive the reference
            if label in pretrained.labels_spec_["regular"]:
                derivation_type = "regular"
            elif label in pretrained.labels_spec_["union"]:
                derivation_type = "union"
            elif label in pretrained.labels_spec_["intersection"]:
                derivation_type = "intersection"
            else:
                raise ValueError("%s not found in training labels : %s"
                                 % (label, pretrained.label_spec_))

            # Load pipeline metric (when available)
            try:
                metric = pipeline.get_metric()
            except NotImplementedError as e:
                metric = None


            # Apply pipeline and dump output to RTTM files
            output_rttm = output_dir / f'{protocol_name}.{subset}.{label}.rttm'
            with open(output_rttm, 'w') as fp:
                for current_file in getattr(protocol, subset)():
                    hypothesis = pipeline(current_file)
                    pipeline.write_rttm(fp, hypothesis)

                    # compute evaluation metric (when possible)
                    if 'annotation' not in current_file:
                        metric = None

                    # compute evaluation metric (when available)
                    if metric is None:
                        continue

                    reference = current_file['annotation']
                    if derivation_type == "regular":
                        reference = reference.subset([label])
                    else:
                        reference = MultilabelTask.derives_label(reference,
                                                                 derivation_type=derivation_type,
                                                                 meta_label=label,
                                                                 regular_labels=pretrained.
                                                                 labels_spec_[derivation_type][label])
                    uem = get_annotated(current_file)
                    _ = metric(reference, hypothesis, uem=uem)

            # we continue looping through classes, even though metric is not define
            if metric is None:
                continue

            output_eval = output_dir / f'{protocol_name}.{subset}.{label}.eval'
            with open(output_eval, 'w') as fp:
                fp.write(str(metric))

