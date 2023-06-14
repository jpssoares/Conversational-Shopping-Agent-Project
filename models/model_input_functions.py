class TypicalTransformerEncoderQAInputFunction:
    def __init__(self, max_sequence_len: int, to_prepend: str, **kwargs):
        self.max_sequence_len = max_sequence_len
        self.to_prepend = to_prepend
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)
        try:
            if self.slot_filling_to_prepend:
                self.to_prepend = self.slot_filling_to_prepend
        except AttributeError:
            pass

    def __call__(
        self,
        dataset=None,
        item=None,
        index=None,
        question="",
        context="",
        tokenizer=None,
        **kwargs,
    ):
        if dataset is not None:
            tokenizer = dataset.tokenizer
            question = item.question
            context = item.context

        return self.typical_transformer_encoder_qa_input_function(
            tokenizer, question, context, **kwargs
        )

    def typical_transformer_encoder_qa_input_function(
        self, tokenizer, question, context, **kwargs
    ):
        tokenizer_out = tokenizer(
            self.to_prepend + question,
            context,
            max_length=self.max_sequence_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        return tokenizer_out


class TypicalTransformerEncoderQAInputFunctionWithExtraSlotInfo:
    def __init__(self, max_sequence_len: int, to_prepend: str, **kwargs):
        import json

        with open(
            f"./eval/dst_eval/eval-data/multiwoz21/validation.json", "r"
        ) as json_data_eval:
            self.data_eval = json.load(json_data_eval)
        with open(
            f"./eval/dst_eval/eval-data/multiwoz21/test.json", "r"
        ) as json_data_test:
            self.data_test = json.load(json_data_test)
        self.max_sequence_len = max_sequence_len
        self.to_prepend = to_prepend
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    def __call__(
        self,
        dataset=None,
        item=None,
        index=None,
        question="",
        context="",
        tokenizer=None,
        **kwargs,
    ):
        import models.statetrackingutils as statetrackingutils

        if dataset is not None:
            tokenizer = dataset.tokenizer
            question = item.question
            context = item.context
        final_str = ""
        all_filled_slots = dict()
        inp_func = self.typical_transformer_encoder_qa_input_function(
            tokenizer, question, final_str + " " + context, **kwargs
        )
        if item.utterance_index > 0 and dataset.split != "train":
            if dataset.split == "test":
                vals = self.data_test[item.dialogue_id][str(item.utterance_index - 1)][
                    "turn_belief"
                ]
            else:
                vals = self.data_eval[item.dialogue_id][str(item.utterance_index - 1)][
                    "turn_belief"
                ]

            for prev_slot_info in vals:
                k, v = prev_slot_info.split("~")
                all_filled_slots[k] = v
            all_filled_slots.pop(item.slot, None)

            all_filled_slots = dict(sorted(all_filled_slots.items()))
            for k, v in all_filled_slots.items():
                final_str += " the " + k.replace("-", " ") + " is " + v + ". "

            inp_func = self.typical_transformer_encoder_qa_input_function(
                tokenizer, question, final_str + " " + context, **kwargs
            )
            if item.answer and item.answer != [""]:
                s, e = statetrackingutils.find_span(
                    item.answer, inp_func.encodings[0].tokens
                )
                if s != -1 and e != -1:
                    item.start_position = s
                    item.end_position = e
            # if item.answer != ['']:
            #    print(inp_func.encodings[0].tokens, item.start_position, item.end_position, item.answer)
        return inp_func

    def typical_transformer_encoder_qa_input_function(
        self, tokenizer, question, context, **kwargs
    ):
        tokenizer_out = tokenizer(
            self.to_prepend + question,
            context,
            max_length=self.max_sequence_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        return tokenizer_out


class T5PPromptedInputFunction:
    def __init__(
        self, max_sequence_len: int, to_prepend_a: str, to_prepend_b: str, **kwargs
    ):
        self.max_sequence_len = max_sequence_len
        self.to_prepend_a = to_prepend_a
        self.to_prepend_b = to_prepend_b
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    def __call__(
        self,
        dataset=None,
        item=None,
        index=None,
        question="",
        context="",
        tokenizer=None,
        **kwargs,
    ):
        if dataset is not None:
            tokenizer = dataset.tokenizer
            question = item.question
            context = item.context
        return self.typical_transformer_encoder_qa_input_function(
            tokenizer, question, context, **kwargs
        )

    def typical_transformer_encoder_qa_input_function(
        self, tokenizer, question, context, **kwargs
    ):
        tokenizer_out = tokenizer(
            self.to_prepend_a + question + " " + self.to_prepend_b + context,
            max_length=self.max_sequence_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        return tokenizer_out
