import json
import os
import collections
import numpy as np
from torch.utils.data import DataLoader, Dataset
from itertools import chain
import torch
from random import shuffle
from numpy.random import rand

import source.statetrackingutils as statetrackingutils
import source.config as config
from collections.abc import MutableSequence
from copy import deepcopy

class SFDataset(Dataset):
    def __init__(
        self,
        loc="",
        name="",
        split="",
        tokenizer=None,
        max_sequence_len=180,
        toy=False,
        force_len=-1,
        shuffle_samples=True,
        create_empty=False,
        using_system_utterances=True,
        dropout=0,
        context_window_length=0,
        negative_example_proportion=1,
        model_input_function=None,
    ):
        self.loc = loc
        self.name = name
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.toy = toy
        self.force_len = force_len
        self.shuffle_samples = shuffle_samples
        self.dropout = dropout
        self.context_window_length = context_window_length
        self.using_system_utterances = using_system_utterances
        # extra_info: { slots, tokens, utterance_index, dialogue_id }
        self.extra_info = []
        self.all_items = []
        self.all_negative_items = []
        self.all_negative_items_to_sample = []
        self.all_intents = []
        self.all_slots = []
        self.categorical_slots = []
        self.all_items_correspondence = []
        self.all_negative_items_correspondence = []
        self.all_negative_items_to_sample_correspondence = []
        self.negative_example_proportion = negative_example_proportion
        self.label_names = ["labels", "start_positions", "end_positions"]
        self.fixed_params = ["dropout"]
        self.model_input_function = model_input_function
        self.user_token = model_input_function.user_token
        self.system_token = model_input_function.system_token

        if create_empty:
            return

    def __len__(self):
        # TODO rename extra_info -> all_items
        return len(self.extra_info) + len(self.all_negative_items_to_sample)

    def _is_positive_example(self, i):
        return i < len(self.extra_info)

    def __getitem__(self, item):
        item_to_return = deepcopy(self.get_item(item).input_info)
        if self.dropout > 0 and self._is_positive_example(item):
            self.handle_slot_dropout(self.get_item(item), item_to_return["input_ids"])
        return item_to_return, item


    def _is_list_or_listlike(self, item):
        return isinstance(item, (MutableSequence, dict, np.ndarray, torch.Tensor))

    def get_item(self, item):
        ret = []
        if self._is_list_or_listlike(item):
            if isinstance(item, torch.Tensor) and len(item.shape) == 0:
                item = item.item()
            else:
                for i in item:
                    ret.append(self.get_item(i))
                return ret

        if not self._is_positive_example(item):
            item_to_return = self.all_negative_items_to_sample[item - len(self.all_items)]
        else:
            item_to_return = self.extra_info[item]
        return item_to_return

    def handle_slot_dropout(self, item: statetrackingutils.SlotFillingInputInfo, input_ids): # TODO this could be made more complicated to promote more input variations
        # could be interesting to have all slots mentioned in this utterance be potentially dropped out
        for single_item in item.slots.values():
            r = rand(single_item.span_end - single_item.span_start + 1)
            for i, random_i in zip(range(single_item.span_start, single_item.span_end + 1), r):
                if random_i < self.dropout:
                    input_ids[i] = self.tokenizer.mask_token_id
