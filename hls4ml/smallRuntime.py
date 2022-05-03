from __future__ import annotations
from tndm.utils.args_parser import ArgsParser
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union
from collections.abc import Iterable
import numpy as np
import json
import yaml
import os
import pandas as pd
from sklearn.linear_model import Ridge
from datetime import datetime
import getpass
import socket

from tndm import TNDM, LFADS
from tndm.utils import AdaptiveWeights, logger, CustomEncoder, LearningRateStopping
import sys
sys.path.insert(1, '../tndm')
from parser import Parser, ModelType

tf.config.run_functions_eagerly(True)


class Runtime(object):

    @staticmethod
    def clean_datasets(
            train_dataset: Union[List[tf.Tensor], tf.Tensor],
            val_dataset: Optional[Union[List[tf.Tensor], tf.Tensor]] = None,
            with_behaviour: bool = False) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Optional[Tuple[tf.Tensor, tf.Tensor]]]:

        train_neural = None
        train_behaviour = None
        valid = None

        if tf.debugging.is_numeric_tensor(train_dataset) or (
                isinstance(train_dataset, np.ndarray)):
            train_dataset = (train_dataset,)
        elif isinstance(train_dataset, Iterable):
            train_dataset = tuple(train_dataset)

        if tf.debugging.is_numeric_tensor(val_dataset) or (
                isinstance(val_dataset, np.ndarray)):
            val_dataset = (val_dataset,)
        elif isinstance(val_dataset, Iterable):
            val_dataset = tuple(val_dataset)

        if with_behaviour:
            assert len(train_dataset) > 1, ValueError(
                'The train dataset must be a list containing two elements: neural activity and behaviour')
            neural_dims = train_dataset[0].shape[1:]
            behavioural_dims = train_dataset[1].shape[1:]
            train_neural, train_behaviour = train_dataset[:2]
            if val_dataset is not None:
                if len(val_dataset) > 1:
                    assert neural_dims == val_dataset[0].shape[1:], ValueError(
                        'Validation and training datasets must have coherent sizes')
                    assert behavioural_dims == val_dataset[1].shape[1:], ValueError(
                        'Validation and training datasets must have coherent sizes')
                    valid = val_dataset[:2]
        else:
            assert len(train_dataset) > 0, ValueError(
                'Please provide a non-empty train dataset')
            neural_dims = train_dataset[0].shape[1:]
            train_neural, train_behaviour = train_dataset[0], None
            if val_dataset is not None:
                assert neural_dims == val_dataset[0].shape[1:], ValueError(
                    'Validation and training datasets must have coherent sizes')
                valid = (val_dataset[0], None)

        return (train_neural, train_behaviour), valid

    @staticmethod
    def train(modeldir, model_type: Union[str, ModelType], model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int,
              train_dataset: Tuple[tf.Tensor, tf.Tensor], adaptive_weights: AdaptiveWeights,
              val_dataset: Optional[Tuple[tf.Tensor, tf.Tensor]] = None, batch_size: Optional[int] = None, logdir: Optional[str] = None,
              adaptive_lr: Optional[Union[dict, tf.keras.callbacks.Callback]] = None, layers_settings: Dict[str, Any] = {},
              terminating_lr: Optional[float]=None, verbose: Optional[int] = 2):

        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)

        if layers_settings is None:
            layers_settings = {}

        (x, y), validation_data = Runtime.clean_datasets(
            train_dataset, val_dataset, model_type.with_behaviour)

        if model_type == ModelType.TNDM:
            model_settings.update(
                neural_dim=x.shape[-1],
                behaviour_dim=y.shape[-1],
            )
            model = TNDM(
                **model_settings,
                layers=layers_settings
            )
        elif model_type == ModelType.LFADS:
            model_settings.update(
                neural_dim=x.shape[-1],
            )
            model = LFADS(
                **model_settings,
                layers=layers_settings
            )
        else:
            raise NotImplementedError(
                'This model type has not been implemented yet')

        callbacks = [adaptive_weights]
        if logdir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))
        if adaptive_lr is not None:
            if isinstance(adaptive_lr, tf.keras.callbacks.Callback):
                callbacks.append(adaptive_lr)
            else:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss/reconstruction', **adaptive_lr))
        if terminating_lr is not None:
            callbacks.append(LearningRateStopping(terminating_lr))

        model.build(input_shape=[None] + list(x.shape[1:]))

        model.compile(
            optimizer=optimizer,
            loss_weights=adaptive_weights.w
        )

        try:
            history = model.fit(
                x=x,
                y=y,
                callbacks=callbacks,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=verbose
            )
        except KeyboardInterrupt:
            return model, None
        model.save(modeldir)
        model.summary()
        return model, history