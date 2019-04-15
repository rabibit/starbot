from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, List

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from .dataset import create_dataset
from .bert.tokenization import FullTokenizer
from .bert import modeling
from .bert import optimization
import tensorflow as tf
import os

BERT_MODEL_FILE_NAME = "bert_model.pkl"

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", "bert_ner/checkpoint/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", "bert_ner/output/result_dir",
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", "bert_ner/checkpoint/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class BertExtractor(EntityExtractor):
    name = "ner_bert"
    provides = ["entities"]
    input_length = 128
    max_seq_length = 128

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        self.vocab = FullTokenizer('./bert_ner/checkpoint/vocab.txt')  # todo
        self.dataset = create_dataset(training_data.training_examples)
        train_file = "train.tf_record"  # todo
        self._prepare_features(train_file)
        if not FLAGS.do_train:
            raise ValueError("At least one of 'do_train' of 'do_eval' must be True.")
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

        if FLAGS.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, bert_config.max_position_embeddings)
            )

        tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None

        if FLAGS.do_train:
            train_examples = self.dataset.examples
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = self._model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self.dataset.labels) + 1,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

        self.ent_tagger = estimator

        if FLAGS.do_train:
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            train_input_fn = self._file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
            self.ent_tagger.train(input_fn=train_input_fn, max_steps=num_train_steps)

    def _pad(self, lst, v):
        n = self.input_length - len(lst)
        if n > 0:
            return lst + [v] * n
        else:
            return lst

    def _create_int_feature(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=self._pad(list(values), 0)))

    def _prepare_features(self, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for example in self.dataset.examples:
                inputs = [ex.char for ex in example]
                labels = [ex.label for ex in example]
                input_ids = self.vocab.convert_tokens_to_ids(inputs)
                input_mask = [1 for _ in inputs]
                label_ids = self.dataset.label2id(labels)
                seg_ids = [0 for _ in inputs]

                features = {"input_ids": self._create_int_feature(input_ids),
                            "input_mask": self._create_int_feature(input_mask),
                            "segment_ids": self._create_int_feature(seg_ids),
                            "label_ids": self._create_int_feature(label_ids)}
                record = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(record.SerializeToString())

    def _file_based_input_fn_builder(self, input_file, seq_length, is_training, drop_remainder):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_ids":tf.VarLenFeature(tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            batch_size = params["batch_size"]
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            ))
            return d

        return input_fn

    def _create_model(self, bert_config, is_training, input_ids, input_mask,
                      segment_ids, labels, num_labels, use_one_hot_embeddings):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        output_layer = model.get_sequence_output()

        hidden_size = output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )
        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, num_labels])
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=-1)
            return loss, per_example_loss, logits, predict, [output_weight, output_bias]

    def _model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_tpu,
                         use_one_hot_embeddings):
        def _model_fn(features, labels, mode, params):
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            # label_mask = features["label_mask"]
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, predicts, tune_vars) = self._create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)
            tvars = tf.trainable_variables()
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                if use_tpu:
                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.logging.info("**** Trainable Variables ****")

            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            output_spec = None
            tf.trainable_variables = lambda: tune_vars
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
                )
            return output_spec

        return _model_fn

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        # <class 'dict'>:
        # {'start': 5, 'end': 7, 'value': '标间', 'entity': 'room_type',
        # 'confidence': 0.9988710946115964, 'extractor': 'ner_crf'}
        extracted = self.add_extractor_name(self._extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        from sklearn.externals import joblib

        if self.ent_tagger:
            model_file_name = os.path.join(model_dir, BERT_MODEL_FILE_NAME)

            joblib.dump(self.ent_tagger, model_file_name)

        return {"classifier_file": BERT_MODEL_FILE_NAME}

    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> CRFEntityExtractor
        from sklearn.externals import joblib

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", BERT_MODEL_FILE_NAME)
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            ent_tagger = joblib.load(model_file)
            return cls(meta, ent_tagger)
        else:
            return cls(meta)

    def _extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""

        self.vocab = FullTokenizer('./bert_ner/checkpoint/vocab.txt')  # todo
        self.dataset = create_dataset(message.text)
        predict_file = "predict.tf_record"  # todo
        self._prepare_features(predict_file)
        if self.ent_tagger is not None:
            if FLAGS.use_tpu:
                # Warning: According to tpu_estimator.py Prediction on TPU is an
                # experimental feature and hence not supported here
                raise ValueError("Prediction in TPU not supported")
            predict_drop_remainder = True if FLAGS.use_tpu else False


            predict_input_fn = self._file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)

            result = self.ent_tagger.predict(input_fn=predict_input_fn)
            return result
        else:
            return []
