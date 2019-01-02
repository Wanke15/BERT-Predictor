#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import dot
from numpy.linalg import norm
import numpy as np

import modeling
import tokenization
import tensorflow as tf

from extract_features import InputExample
from extract_features import convert_examples_to_features
from extract_features import model_fn_builder


def read_examples(string1, string2):
    """Read a list of `InputExample`s from two given strings."""
    examples = []
    unique_id = 0

    text_a = tokenization.convert_to_unicode(string1)
    text_b = tokenization.convert_to_unicode(string2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


def read_examples_for_encode(string_list):
    """Read a list of `InputExample`s from a given list of strings."""
    examples = []
    unique_id = 0
    for string in string_list:
        text_a = tokenization.convert_to_unicode(string)
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=''))
        unique_id += 1
    return examples


class Model(object):
    def __init__(self, vocab_file, bert_config_file, init_checkpoint,
                 max_seq_length, layer=[-1], pooling_strategy='MEAN',
                 do_lower_case=True, master=None,
                 num_tpu_cores=None, model_dir=None):

        tf.logging.set_verbosity(tf.logging.WARN)

        self.vocab_file = vocab_file
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.layer = layer
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.use_tpu = False
        self.use_one_hot_embeddings = False
        self.master = master
        self.num_tpu_cores = num_tpu_cores
        self.model_dir = model_dir
        self.estimator = None
        self.predictor = None
        self.pooling_strategy = pooling_strategy
        assert self.pooling_strategy in ('MEAN', 'MAX'), "Unknown pooling strategy!!!"

        self._load()

    def _load(self):

        self.layer_indexes = [l for l in self.layer]
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        self.model_fn = model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.init_checkpoint,
            layer_indexes=self.layer_indexes,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            master=self.master,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=self.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.use_tpu,
            model_fn=self.model_fn,
            config=run_config,
            predict_batch_size=None,
            model_dir=self.model_dir)

        def serving_input_fn():
            '''
            serving input function for 'tf.contrib.predictor'
            :return: instance of 'tf.estimator.export.ServingInputReceiver'
            '''
            num_examples = None
            seq_length = self.max_seq_length

            pred_input = {
                "unique_ids":
                    tf.placeholder(shape=[num_examples], dtype=tf.int32),
                "input_ids":
                    tf.placeholder(
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.placeholder(
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_type_ids":
                    tf.placeholder(
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
            }
            return tf.estimator.export.ServingInputReceiver(pred_input, pred_input)

        self.predictor = tf.contrib.predictor.from_estimator(self.estimator, serving_input_fn)

    def _get_similairty(self):
        '''For single layer'''
        '''
        vec1 = np.array([vec['layers'][0]['values'] for vec in self.res1['features']])
        vec2 = np.array([vec['layers'][0]['values'] for vec in self.res2['features']])
        
        vec1 = np.mean(vec1, axis=0)
        vec2 = np.mean(vec2, axis=0)
        '''

        '''For multiple layers'''
        vec1 = np.array([[layer['values'] for layer in vec['layers']] for vec in self.res1['features']])
        vec2 = np.array([[layer['values'] for layer in vec['layers']] for vec in self.res2['features']])

        if self.pooling_strategy == 'MEAN':
            vec1 = np.mean(vec1, axis=(0, 1))
            vec2 = np.mean(vec2, axis=(0, 1))
        elif self.pooling_strategy == 'MAX':
            vec1 = np.max(vec1, axis=(0, 1))
            vec2 = np.max(vec2, axis=(0, 1))
        else:
            raise("Unknown pooling strategy!!!")


        return self._cos_sim(vec1, vec2)

    def _cos_sim(self, a, b):
        '''
        Cosine similarity
        :param a:
        :param b:
        :return: Cosine similarity
        '''
        return dot(a, b) / (norm(a) * norm(b))

    def _predict(self, string):
        '''
        Get encoding for single input string
        :param string:
        :return: BERT output encoding
        '''
        examples = read_examples(string, '')
        features = convert_examples_to_features(examples=examples, seq_length=self.max_seq_length,
                                                tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_input_type_ids = []
        for feature in list(features):
            all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_input_type_ids.append(feature.input_type_ids)

        input_string = {"input_ids": all_input_ids, "unique_ids": all_unique_ids,
                        "input_type_ids": all_input_type_ids, "input_mask": all_input_mask}

        result = self.predictor(input_string)
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        output_json = {}
        output_json["linex_index"] = unique_id
        all_features = []
        for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for (j, layer_index) in enumerate(self.layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layers = {}
                layers["index"] = layer_index
                layers["values"] = [
                    round(float(x), 6) for x in layer_output[0][i:(i + 1)].flat
                ]
                all_layers.append(layers)
            features = {}
            features["token"] = token
            features["layers"] = all_layers
            all_features.append(features)

        output_json["features"] = all_features

        return output_json

    def similarity(self, string1, string2):
        '''
        Get cosine similarity of the two given strings: string1, string2
        :param string1:
        :param string2:
        :return: cosine similarity
        '''
        self.res1 = self._predict(string1)
        self.res2 = self._predict(string2)

        _similarity = self._get_similairty()

        return float('%.4f' % _similarity)

    def encode(self, string_list):
        '''
        Encode a list of string to a list of BERT 768-dim encodings
        :param string_list:
        :return: list(string_encodes)
        '''
        examples = read_examples_for_encode(string_list)
        features = convert_examples_to_features(examples=examples, seq_length=self.max_seq_length,
                                                tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_input_type_ids = []
        for feature in list(features):
            all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_input_type_ids.append(feature.input_type_ids)

        input_string = {"input_ids": all_input_ids, "unique_ids": all_unique_ids,
                        "input_type_ids": all_input_type_ids, "input_mask": all_input_mask}

        all_sentences_encodes = []
        # Make predictions
        result = self.predictor(input_string)
        for unique_id in result["unique_id"]:
            unique_id = int(unique_id)
            feature = unique_id_to_feature[unique_id]
            all_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(self.layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = {}
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output[unique_id][i:(i + 1)].flat
                        # unique_id represents each sentence
                    ]
                    all_layers.append(layers)
                features = {}
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
            # add multi-layer support
            single_sentence_all_tok_encodes = np.array(
                [[layer['values'] for layer in tok_feature['layers']] for tok_feature in all_features])
            if self.pooling_strategy == 'MEAN':
                single_sentence_all_tok_encodes = np.mean(single_sentence_all_tok_encodes, axis=(0, 1))
            elif self.pooling_strategy == 'MAX':
                single_sentence_all_tok_encodes = np.max(single_sentence_all_tok_encodes, axis=(0, 1))

            all_sentences_encodes.append(single_sentence_all_tok_encodes)
        all_sentences_encodes = np.array(all_sentences_encodes)

        return all_sentences_encodes
