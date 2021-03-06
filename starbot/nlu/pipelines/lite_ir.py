import logging
import numpy as np
from pathlib import Path

import tensorflow as tf
from rasa.nlu.training_data import Message
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Activation
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor

from typing import Dict, Text, Any, Optional, List
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import TrainingData
from starbot.nlu.pipelines.bert_embedding.dataset import (LabelMap,
                                                          create_dataset,
                                                          Dataset,
                                                          Sentence)

logger = logging.getLogger(__name__)


def create_model(num_ir_labels: int) -> Sequential:
    model = Sequential()
    model.add(Bidirectional(LSTM(768, return_sequences=False)))
    model.add(Dense(num_ir_labels))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


class LiteClassifier(Component):
    model: Sequential
    ir_labels: LabelMap
    graph: tf.Graph
    session: tf.Session

    requires = ['bert_embedding']

    MODEL_DIR = 'liteir'

    def __init__(self, component_config: Dict[Text, Any]):
        super().__init__(component_config)
        self.input_length = component_config['max_seq_length']
        self.batch_size = component_config['batch_size']
        self.epochs = component_config['epochs']

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> 'Component':
        return super(LiteClassifier, cls).create(component_config, config)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['LiteClassifier'] = None,
             **kwargs: Any
             ) -> 'LiteClassifier':

        if cached_component:
            return cached_component
        else:
            slf = cls(meta)
            slf._prepare_for_prediction(model_dir)
            return slf

    def _prepare_for_prediction(self, model_dir):
        mdir = Path(model_dir) / self.MODEL_DIR
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default(), self.session.as_default():
                self.model = tf.keras.models.load_model(mdir / 'model.h5')
        self.ir_labels = LabelMap.load(mdir / 'ir_labels.json')

    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        dataset = create_dataset(training_data.training_examples)
        self.ir_labels = dataset.intent_labels
        num_ir_labels = len(dataset.intent_labels) + 1

        self.model = create_model(num_ir_labels)

        inputs, labels = zip(*self._prepare_features(dataset))
        inputs = np.array(inputs)
        labels = np.array(labels)

        self.model.fit(inputs, labels, batch_size=self.batch_size, epochs=self.epochs)

    def _create_single_feature_from_message(self, message: Message):
        return message.get('bert_embedding')

    def _create_single_feature(self, example: Sentence, dataset: Dataset):
        embedding = example.message.get('bert_embedding')

        ir_label = example.intent
        ir_label_id = dataset.intent_label2id([ir_label])
        return embedding, ir_label_id

    def _prepare_features(self, dataset: Dataset):
        all_features = []
        for example in dataset.examples:
            all_features.append(self._create_single_feature(example, dataset))
        return all_features

    def process(self, message: Message, **kwargs: Any) -> None:
        ir_label, confidence = self._predict(message)
        ir = {'name': ir_label, 'confidence': confidence}
        message.set("lite_intent", ir, add_to_output=True)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""
        outdir = Path(model_dir) / self.MODEL_DIR
        filename = outdir / 'model.h5'
        filename.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Saving {}'.format(filename))
        self.model.save(filename)

        label_file = outdir / 'ir_labels.json'
        logger.info('Saving {}'.format(label_file))
        self.ir_labels.save(label_file)
        return {}

    def _predict(self, message: Message) -> (str, float):
        """Take a sentence and return entities in json format"""
        features = self._create_single_feature_from_message(message)
        features = np.array([features])
        tf.keras.backend.set_session(self.session)
        with self.graph.as_default():
            logits = self.model.predict(features)
        ir = np.argmax(logits, axis=-1)
        ir_label = self.ir_labels.decode(ir)
        return ir_label[0], logits[0][ir][0].item()
