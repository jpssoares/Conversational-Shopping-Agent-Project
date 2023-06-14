import json
import collections
import numpy as np
import torch
from random import shuffle

from . import SFDataset

import source.statetrackingutils as statetrackingutils
from copy import deepcopy
import source.config as config


class SimXDataset(SFDataset.SFDataset):
    def __init__(
        self,
        loc="",
        name="",
        split="",
        tokenizer=None,
        max_sequence_len=180,
        join_bi=True,
        toy=False,
        force_len=-1,
        shuffle_samples=True,
        create_empty=False,
        using_system_utterances=True,
        dropout=0,
        context_window_length=0,
        negative_example_proportion=1,
        ignore_keys=[],
        ignore_intents=[],
        ignore_domains=[],
        force_domains=[],
        model_input_function=None,
    ):
        super().__init__(
            loc=loc,
            name=name,
            split=split,
            tokenizer=tokenizer,
            max_sequence_len=max_sequence_len,
            toy=toy,
            force_len=force_len,
            shuffle_samples=shuffle_samples,
            create_empty=create_empty,
            using_system_utterances=using_system_utterances,
            dropout=dropout,
            context_window_length=context_window_length,
            negative_example_proportion=negative_example_proportion,
            model_input_function=model_input_function,
        )
        self.loc = loc
        self.name = name
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.join_bi = join_bi
        self.toy = toy
        self.force_len = force_len
        self.shuffle_samples = shuffle_samples
        self.lower_text = True
        self.all_items = []
        self.all_intents = []
        self.all_slots = []
        self.extra_info = []
        # extra_info: { slots, tokens, utterance_index, dialogue_id }
        self.categorical_slots = []
        self.dropout = dropout
        self.context_window_length = context_window_length
        self.label_names = ["labels", "start_positions", "end_positions"]
        self.fixed_params = ["dropout"]
        self.user_token = model_input_function.user_token
        self.system_token = model_input_function.system_token
        try:
            self.to_prepend = model_input_function.slot_filling_to_prepend
        except Exception as e:
            print("SF: Not using slot filling to prepend")
            self.to_prepend = ""
        self.model_input_function = model_input_function
        self.intent_per_utterance = []
        self.using_system_utterances = using_system_utterances
        self.all_items_correspondence = []
        self.ignore_keys = ignore_keys
        self.ignore_intents = ignore_intents
        self.ignore_domains = ignore_domains
        self.force_domains = force_domains  # forced domains are the ONLY ones that are present in the dataset
        self.use_dialogue_state = True
        if create_empty:
            return

        self.all_slots, self.all_intents = set([]), set([])

        with open(loc) as json_dataset:
            data_raw = json.load(json_dataset)

        for d in data_raw:
            for t in d["turns"]:
                for s in t["dialogue_state"]:
                    self.all_slots.add(s["slot"])

        if self.toy and self.split == "train":
            data_raw = data_raw[0:500]
        elif self.toy:
            data_raw = data_raw[0:10]

        elif force_len != -1 and self.split == "train":  # and not self.shuffle_samples:
            # we shuffle the lists to sample N data items, as per "Language Model is All you Need" (qanlu)
            print(
                f"Shuffling data! Old number of dialogues: {len(data_raw)} New number: {int(force_len * len(data_raw))}"
            )
            shuffle(data_raw)
            data_raw = data_raw[0 : int(force_len * len(data_raw))]

        self.remove_topop_slots(data_raw)

        self.all_items, self.extra_info = self.process_data(data_raw)

        self.all_slots = list(self.all_slots)
        self.all_intents = list(self.all_intents)

        # self.generate_outputs_for_dsteval(name, split)

        ignored_intents = set()
        for intent in self.all_intents:
            if any([item in intent.lower() for item in self.ignore_intents]):
                ignored_intents.add(intent)
        self.ignore_intents = list(ignored_intents)

        print(
            f"Slot-filling ({self.name}) dataset loaded... Using state: {self.use_dialogue_state}; Ignoring slots {self.ignore_keys}; Ignoring intents: {self.ignore_intents}; {len(self)} examples.",
            flush=True,
            end="\t",
        )
        if self.split == "train":
            print("All slots:", self.all_slots)
        if self.split == "test":
            config.ignore_keys_validate_test = self.ignore_keys
        print(
            f"Note that it doesn't actually ignore any slot keys and instead simply passes them on to the model (or dataset transformation). They should ignore them instead.",
            flush=True,
        )
        print(
            f"note 2: slot_dict = slot_dict_dialogue_state used to be slot_dict = deepcopy(slot_dict_dialogue_state)",
            flush=True,
        )

    def generate_outputs_for_dsteval(self, name, split):
        import json
        import re
        from copy import deepcopy

        out_file = dict()
        prev = None
        for item in self.extra_info:
            if str(item.dialogue_id) not in out_file:
                out_file[str(item.dialogue_id)] = dict()
            if item.utterance_index > 0:
                out_file[str(item.dialogue_id)][str(item.utterance_index)] = deepcopy(
                    prev
                )
            else:
                out_file[str(item.dialogue_id)][str(item.utterance_index)] = {
                    "turn_belief": dict()
                }
            for slot in item.slots.values():
                if slot.slot_gate == 1:
                    out_file[str(item.dialogue_id)][str(item.utterance_index)][
                        "turn_belief"
                    ][slot.slot_key] = "dontcare"

                elif slot.slot_gate == 2:
                    slot_value = slot.slot_value

                    timepat = re.compile(
                        r"(\d{1,2} : \d{1,2})|(\d{1,2}( )?[.] \d{1,2})"
                    )  # unfortunately we need to do this for trade+mwoz compatibility (for now)

                    def time_repl(matchobj):
                        time = matchobj.string[
                            matchobj.regs[0][0] : matchobj.regs[0][1]
                        ]
                        return time.replace(" ", "")

                    slot_value = re.sub(timepat, time_repl, slot_value)

                    out_file[str(item.dialogue_id)][str(item.utterance_index)][
                        "turn_belief"
                    ][slot.slot_key] = slot_value
            prev = deepcopy(out_file[str(item.dialogue_id)][str(item.utterance_index)])
            out_file[str(item.dialogue_id)][str(item.utterance_index)][
                "turn_belief"
            ] = [
                k + "~" + v
                for k, v in out_file[str(item.dialogue_id)][str(item.utterance_index)][
                    "turn_belief"
                ].items()
            ]
        with open(name + "_" + split + ".json", "w") as out:
            json.dump(out_file, out, indent=4)
        if split == "test":
            with open(name + "_slots.txt", "w") as slots_out:
                for slot in self.all_slots:
                    slots_out.write(slot + "\n")
            print("All splits loaded!", flush=True)
            exit()

    def remove_topop_slots(self, data_raw):
        print("here")
        actual_ignored_keys = set([])
        for dialogue in data_raw:
            for turn in dialogue["turns"]:
                if self.use_dialogue_state:
                    for i in range(len(turn["dialogue_state"])):
                        for ignore_slot in self.ignore_keys:
                            if ignore_slot in turn["dialogue_state"][i]["slot"]:
                                actual_ignored_keys.add(
                                    turn["dialogue_state"][i]["slot"]
                                )
                else:
                    for i in range(len(turn["user_utterance"]["slots"])):
                        for ignore_slot in self.ignore_keys:
                            if (
                                ignore_slot
                                in turn["user_utterance"]["slots"][i]["slot"]
                            ):
                                actual_ignored_keys.add(
                                    turn["user_utterance"]["slots"][i]["slot"]
                                )
                    for i in range(
                        len(turn.get("system_utterance", {}).get("slots", []))
                    ):
                        for ignore_slot in self.ignore_keys:
                            if (
                                ignore_slot
                                in turn["system_utterance"]["slots"][i]["slot"]
                            ):
                                actual_ignored_keys.add(
                                    turn["system_utterance"]["slots"][i]["slot"]
                                )
        self.ignore_keys = list(actual_ignored_keys)

    def _get_full_context(self, prev_state):
        context_window = ""
        for i in range(self.context_window_length):
            if "entire_utterance" not in prev_state:
                break
            context_window = prev_state["entire_utterance"] + " " + context_window
            if "prev_state" in prev_state:
                prev_state = prev_state["prev_state"]

        return context_window

    def process_data(self, data_raw):
        all_data = collections.deque()
        all_extra_info = collections.deque()
        total_i = 0
        slot_dict_dialogue_state = dict()
        for dialogue in data_raw:
            prev_state = {"dialogue_state": {}}
            # if any([item in self.ignore_domains for item in dialogue['services']]): # we remove dialogues in the domains we dont care about (for leave-one-out, etc)
            # continue
            # the previous condition is correct, but per trade,t5-dst,recent google papers:
            if (
                "services" in dialogue
                and self.ignore_domains == dialogue["services"]
                and dialogue["services"]
            ):
                continue
            if len(self.force_domains) > 0 and not any(
                [item in self.force_domains for item in dialogue["services"]]
            ):  # if we're forcing domains and none of the domains are present
                continue
            for i, turn in enumerate(dialogue["turns"]):
                slot_dict = dict()
                user_info = turn["user_utterance"]
                system_info = turn.get("system_utterance")
                user_turn = self.user_token + " " + user_info["text"]
                system_turn = (
                    self.system_token + " " + system_info["text"]
                    if system_info is not None
                    else ""
                )
                entire_utterance = system_turn + " " + user_turn
                context = self._get_full_context(prev_state) if i > 0 else ""
                if self.lower_text:
                    context, entire_utterance = (
                        context.lower(),
                        entire_utterance.lower(),
                    )
                tokenizer_out = self.tokenizer(
                    self.to_prepend + " " + context + " " + entire_utterance,
                    max_length=self.max_sequence_len,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                data_item = tokenizer_out.data
                data_item = {
                    key: torch.squeeze(value) for key, value in data_item.items()
                }
                intent = turn.get("user_intents", ["none"])[0]
                self.all_intents.add(intent)
                data_item["label"] = intent
                tokenized_text = tokenizer_out.encodings[0].tokens
                if self.use_dialogue_state:
                    slot_dict_dialogue_state = self.extract_dontcares(
                        turn["dialogue_state"],
                        prev_state["dialogue_state"],
                        slot_dict_dialogue_state,
                    )
                    slot_dict_dialogue_state = (
                        self.extract_slot_info_from_dialogue_state(
                            slot_dict_dialogue_state,
                            prev_state["dialogue_state"],
                            tokenized_text,
                            turn["dialogue_state"],
                        )
                    )
                    slot_dict_dialogue_state = self.extract_nones(
                        slot_dict_dialogue_state
                    )
                    slot_dict = slot_dict_dialogue_state
                    slot_dict_dialogue_state = dict()
                else:
                    slot_dict = self.extract_dontcares(
                        turn["dialogue_state"], prev_state["dialogue_state"], slot_dict
                    )
                    if system_info is not None:
                        slot_dict = self.extract_slot_info(
                            slot_dict, tokenized_text, system_info
                        )
                    slot_dict = self.extract_slot_info(
                        slot_dict, tokenized_text, user_info
                    )
                    slot_dict = self.extract_nones(slot_dict)

                self.intent_per_utterance.append(intent.lower())
                extra_info = statetrackingutils.SlotFillingInputInfo(
                    slots=slot_dict,
                    tokens=tokenized_text,
                    utterance_index=i,
                    dialogue_id=dialogue["dialogue_id"],
                    intent=intent.lower(),
                    input_info=data_item,
                )
                prev_state["prev_state"] = deepcopy(prev_state)
                prev_state["dialogue_state"] = turn["dialogue_state"]
                prev_state["entire_utterance"] = entire_utterance
                extra_info.previous_state = prev_state[
                    "prev_state"
                ]  # TODO previous state should just be the previous turn's slotfillinginputinfo

                all_extra_info.append(extra_info)
                self.all_items_correspondence.append(total_i)
                total_i += 1

        return list(all_data), np.array(all_extra_info)

    def _slot_in_state(self, slot, state):
        slot_key, slot_value = slot["slot"], slot["value"]
        for item in state:
            if item["slot"] == slot_key and item["value"] == slot_value:
                return True
        return False

    def extract_slot_info_from_dialogue_state(
        self, slot_dict, prev_state, tokenized_text, dialogue_state
    ):
        for slot in dialogue_state:
            try:
                tokenized_slot_value = self.tokenizer.tokenize(
                    " " + slot["value"].lower()
                )
                span_start, span_end = statetrackingutils.find_span(
                    tokenized_text,
                    tokenized_slot_value,
                    end_token=self.tokenizer.pad_token,
                )
            except Exception as e:
                print(slot, dialogue_state, tokenized_text, tokenized_slot_value)
            # print(slot['value'], tokenized_slot_value, self.tokenizer.convert_tokens_to_string(tokenized_slot_value).strip())
            if self._slot_in_state(slot, prev_state) and (
                span_start == -1 and span_end == -1
            ):
                continue
            elif (
                span_start == -1
                and span_end == -1
                and slot["slot"] in slot_dict
                and slot_dict[slot["slot"]].slot_gate == 1
            ):
                # slot isnt present in input (...and we see it's a dontcare)
                continue
            else:
                if span_start == -1:
                    span_start = 0
                    span_end = 0

                slot_dict[slot["slot"]] = statetrackingutils.SingleSlotInputExample(
                    slot_key=slot["slot"],
                    slot_value=self.tokenizer.convert_tokens_to_string(
                        tokenized_slot_value
                    ).strip(),
                    tokenized_slot_value=tokenized_slot_value,
                    slot_gate=2,
                    slot_gate_str="value",
                    start=span_start,
                    end=span_end,
                )
        return slot_dict

    def extract_slot_info(self, slot_dict, tokenized_text, info):
        for slot in info["slots"]:
            if "value" in slot:
                tokenized_slot_value = self.tokenizer.tokenize(slot["value"])
            else:
                tokenized_slot_value: list[str] = self.tokenizer.tokenize(
                    " ".join(
                        info["tokens"][slot["start"] : slot["exclusive_end"]]
                    )  # TODO this might not generalize
                    # but when we tried to run it with a BART tokenizer, things failed, because it turned 6:00 pm into 6:00pm
                )
                if (
                    all([len(item) == 1 for item in tokenized_slot_value])
                    and len(tokenized_slot_value) > 1
                ):
                    # Common error: 6:00 pm -> [6, :, 0, 0, p, m]
                    print(
                        "If this message is flooding the console, you have some tokenization error."
                    )

            span_start, span_end = statetrackingutils.find_span(
                tokenized_text, tokenized_slot_value, end_token="[PAD]"
            )
            if span_start == -1:
                span_start = 0
                span_end = 0
            slot_dict[slot["slot"]] = statetrackingutils.SingleSlotInputExample(
                slot_key=slot["slot"],
                slot_value=self.tokenizer.convert_tokens_to_string(
                    tokenized_slot_value
                ),
                tokenized_slot_value=tokenized_slot_value,
                slot_gate=2,
                slot_gate_str="value",
                start=span_start,
                end=span_end,
            )

        return slot_dict

    def extract_dontcares(self, dialogue_state, prev_state, slot_dict):
        for slot_type in dialogue_state:
            if slot_type["value"] == "dontcare" and (
                len([item for item in prev_state if item["slot"] == slot_type["slot"]])
                == 0
                or not [
                    item for item in prev_state if item["slot"] == slot_type["slot"]
                ][0]["value"]
                == "dontcare"
            ):
                slot_dict[
                    slot_type["slot"]
                ] = statetrackingutils.SingleSlotInputExample(
                    slot_key=slot_type["slot"],
                    slot_value="dontcare",
                    tokenized_slot_value=self.tokenizer.tokenize("dontcare"),
                    slot_gate=1,
                    slot_gate_str="dontcare",
                    start=0,
                    end=0,
                )
        return slot_dict

    def extract_nones(self, slot_dict):
        for slot in self.all_slots:
            if slot not in slot_dict:
                slot_dict[slot] = statetrackingutils.SingleSlotInputExample(
                    slot_key=slot,
                    slot_value="",
                    tokenized_slot_value=[""],
                    slot_gate=0,
                    slot_gate_str="none",
                    start=0,
                    end=0,
                )
        return slot_dict
