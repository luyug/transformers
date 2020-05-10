# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import json

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def ir_convert_examples_to_features(
        examples,
        tokenizer,
        qry_max_length=512,
        psg_max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = IRProcessor()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = 'classification'
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        qry_attention_mask, qry_input_ids, label, qry_token_type_ids = create_exp_ftr(
            ex_index, example, True, label_map, mask_padding_with_zero, qry_max_length, output_mode, pad_on_left,
            pad_token, pad_token_segment_id, tokenizer)

        psg_attention_mask, psg_input_ids, label, psg_token_type_ids = create_exp_ftr(
            ex_index, example, False, label_map, mask_padding_with_zero, psg_max_length, output_mode, pad_on_left,
            pad_token, pad_token_segment_id, tokenizer)

        features.append(
            {
                'query': InputFeatures(
                    input_ids=qry_input_ids,
                    attention_mask=qry_attention_mask,
                    token_type_ids=qry_token_type_ids,
                    label=label
                ),
                'passage': InputFeatures(
                    input_ids=psg_input_ids,
                    attention_mask=psg_attention_mask,
                    token_type_ids=psg_token_type_ids,
                    label=label
                ),

            }
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


def create_exp_ftr(ex_index, example, is_query, label_map, mask_padding_with_zero, max_length, output_mode, pad_on_left,
                   pad_token, pad_token_segment_id, tokenizer):
    inputs = tokenizer.encode_plus(example.text_a if is_query else example.text_b,
                                   None, add_special_tokens=True, max_length=max_length, )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
    )
    if output_mode == "classification":
        label = label_map[example.label]
    elif output_mode == "regression":
        label = float(example.label)
    else:
        raise KeyError(output_mode)
    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        logger.info("label: %s (id = %d)" % (example.label, label))
    return attention_mask, input_ids, label, token_type_ids


class IRProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["query"].numpy().decode("utf-8"),
            tensor_dict["document"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ClueWebProcessor(DataProcessor):

    def __init__(self, fold=0, query_field='desc'):
        self.max_test_depth = 100
        self.max_train_depth = 100
        self.n_folds = 5
        self.fold = fold
        self.q_fields = query_field.split(' ')
        logging.info("Using query fields {}".format(' '.join(self.q_fields)))

        self.train_folds = [(self.fold + i) % self.n_folds + 1 for i in range(self.n_folds - 1)]
        self.test_folds = (self.fold + self.n_folds - 1) % self.n_folds + 1
        logging.info("Train Folds: {}".format(str(self.train_folds)))
        logging.info("Test Fold: {}".format(str(self.test_folds)))

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["query"].numpy().decode("utf-8"),
            tensor_dict["document"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        examples = []
        train_files = ["{}.trec.with_json".format(i) for i in self.train_folds]

        qrel_file = open(os.path.join(data_dir, "qrels"), 'r')
        qrels = self._read_qrel(qrel_file)
        logging.info("Qrel size: {}".format(len(qrels)))

        query_file = open(os.path.join(data_dir, "queries.json"), 'r')
        qid2queries = self._read_queries(query_file)
        logging.info("Loaded {} queries.".format(len(qid2queries)))

        for file_name in train_files:
            train_file = open(os.path.join(data_dir, file_name), 'r')
            for i, line in enumerate(train_file):
                items = line.strip().split('#')
                trec_line = items[0]

                qid, _, docid, r, _, _ = trec_line.strip().split(' ')
                assert qid in qid2queries, "QID {} not found".format(qid)
                q_json_dict = qid2queries[qid]
                q_text_list = [q_json_dict[field] for field in self.q_fields]

                json_dict = json.loads('#'.join(items[1:]))
                body_words = json_dict["doc"]["body"].split(' ')
                truncated_body = ' '.join(body_words[0: min(200, len(body_words))])
                d = json_dict["doc"].get("title", "") + ". " + truncated_body

                r = int(r)
                if r > self.max_train_depth:
                    continue
                label = "0"
                if (qid, docid) in qrels or (qid, docid.split('_')[0]) in qrels:
                    label = "1"
                guid = "train-%s-%s" % (qid, docid)
                examples.append(
                    InputExample(guid=guid, text_a=' '.join(q_text_list), text_b=d, label=label)
                )
            train_file.close()
        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        dev_file = open(os.path.join(data_dir, "{}.trec.with_json".format(self.test_folds)), 'r')
        qrel_file = open(os.path.join(data_dir, "qrels"), 'r')
        qrels = self._read_qrel(qrel_file)
        logging.info("Qrel size: {}".format(len(qrels)))

        query_file = open(os.path.join(data_dir, "queries.json"), 'r')
        qid2queries = self._read_queries(query_file)
        logging.info("Loaded {} queries.".format(len(qid2queries)))

        for i, line in enumerate(dev_file):
            items = line.strip().split('#')
            trec_line = items[0]

            qid, _, docid, r, _, _ = trec_line.strip().split(' ')
            assert qid in qid2queries, "QID {} not found".format(qid)
            q_json_dict = qid2queries[qid]
            q_text_list = [q_json_dict[field] for field in self.q_fields]

            json_dict = json.loads('#'.join(items[1:]))
            body_words = json_dict["doc"]["body"].split(' ')
            truncated_body = ' '.join(body_words[0: min(450, len(body_words))])

            # we use the concatentation of title and document first 200 tokens
            d = json_dict["doc"].get("title", "") + ". " + truncated_body

            r = int(r)
            if r > self.max_test_depth:
                continue
            label = "0"
            if (qid, docid) in qrels or (qid, docid.split('_')[0]) in qrels:
                label = "1"
            guid = "test-%s-%s" % (qid, docid)
            examples.append(
                InputExample(guid=guid, text_a=' '.join(q_text_list), text_b=d, label=label)
            )
        dev_file.close()
        return examples

    def _read_qrel(self, qrel_file):
        qrels = set()
        for line in qrel_file:
            qid, _, docid, rel = line.strip().split(' ')
            rel = int(rel)
            if rel > 0:
                qrels.add((qid, docid))
        return qrels

    def _read_queries(self, query_file):
        qid2queries = {}
        for i, line in enumerate(query_file):
            json_dict = json.loads(line)
            qid = json_dict['qid']
            qid2queries[qid] = json_dict
            if i < 3:
                logging.info("Example Q: {}".format(json_dict))
        return qid2queries

    def get_labels(self):
        return ["0", "1"]


class FMarcoProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["query"].numpy().decode("utf-8"),
            tensor_dict["document"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


ir_tasks_num_labels = {
    "marco-2-seg": 2,
    "cw": 2,
    "marco": 2,
}

ir_processors = {
    "marco-2-seg": IRProcessor,
    "cw": ClueWebProcessor,
    "marco": FMarcoProcessor,
}

ir_output_modes = {
    "marco-2-seg": "classification",
    "cw": "classification",
    "marco": "classification",
}
