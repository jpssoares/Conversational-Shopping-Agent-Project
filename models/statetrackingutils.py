from typing import Tuple, Optional, Union
import abc


class ModelOutput:
    def __init__(self, value, identifier=None, *args, **kwargs):
        self.value = value
        self.identifier = identifier

    def set_identifier(self, identifier):
        self.identifier = identifier


class SlotFillingIntentDetectionModelOutput(ModelOutput):
    # the "main" value is the slot filling output
    def __init__(
            self,
            slot_filling_values=None,
            intent=None,
            dialogue="",
            utterance=-1,
            tokens=None,
            *args,
            **kwargs
    ):
        """
        About intents: Some datasets may have more than one user intent per utterance.
        Intent = None -> model or dataset doesn't employ intent detection
        Intent = [] -> model or dataset employs intent detection but it hasn't been set yet
        """
        super().__init__(slot_filling_values if slot_filling_values is not None else {})
        self.intents = intent
        self.dialogue = dialogue
        self.utterance = utterance
        self.tokens = tokens

    def __str__(self):
        return f'{self.dialogue}, {self.utterance}, Intent: {self.intents}; Slots: {[(key, value) for key, value in self.value.items()]}'

    def __repr__(self):
        return str(self)

    def is_empty(self):
        return self.value == {} and not self.intents

    def set_dialogue_utterance(self, dialogue, utterance):
        super().set_identifier(str(dialogue) + "~" + str(utterance))
        self.dialogue = dialogue
        self.utterance = utterance

    def has_intent_detection(self):
        return self.intents is not None

    def get_slot_value_from_key(self, key):
        return self.value[key]

    def get_slot_keys(self):
        return self.value.keys()

    def set_slot(self, key, value):
        self.value[key] = value

    def set_intent(self, intent):
        self.intents = [intent]

    def add_intent(self, intent):
        self.intents.append(intent)

    def get_intent(self) -> str:
        return self.intents[0] if self.intents else 'NONE'

    def set_all_slot_keys(self, all_slot_keys, default_value=""):
        for key in all_slot_keys:
            if key not in self.value:
                self.value[key] = default_value

    def set_tokens(self, tokens):
        self.tokens = tokens


class SingleSlotInputExample:

    def __init__(self, slot_key, slot_value, slot_gate_str, slot_gate,
                 start=-1, end=-1, is_categorical=False, tokenized_slot_value=None, *args, **kwargs):
        self.slot_key: str = slot_key.strip()
        self.slot_value: str = slot_value
        self.tokenized_slot_value: list[str] = tokenized_slot_value
        self.slot_gate_str: str = slot_gate_str
        self.slot_gate: int = slot_gate

        # these are method specific, maybe should just be a subclass or something
        self.span_start: int = start
        self.span_end: int = end
        self.is_categorical: bool = is_categorical

        if self.tokenized_slot_value is None:
            # if the tokenized slot value is not set, we assume that the slot value is already kind of tokenized
            self.tokenized_slot_value = self.slot_value.split(' ')

    def __str__(self) -> str:
        return f'{self.slot_key}: {self.slot_value} ({self.slot_gate_str})'

    def __repr__(self) -> str:
        return str(self)


class InputInfo:
    def __init__(self, input_info, *args, **kwargs):
        self.input_info = input_info  # Generally, this should be enough to a _forward_ call


class SlotFillingInputInfo(InputInfo):
    def __init__(self, slots, tokens, utterance_index, dialogue_id, intent, previous_state=None, input_info=None, *args,
                 **kwargs):
        super().__init__(input_info, *args, **kwargs)
        self.slots: dict[str, SingleSlotInputExample] = slots
        self.tokens: list[str] = tokens
        self.utterance_index: int = utterance_index
        self.dialogue_id: str = dialogue_id
        self.intent: list[str] = intent
        self.previous_state: dict = previous_state

    def __str__(self):
        return f'Utterance {self.utterance_index} of dialogue {self.dialogue_id}: {self.tokens}'

    def __repr__(self):
        return str(self)


class SlotFillingQAInputInfo(InputInfo): # this should extend qainputinfo..
    def __init__(self, slot, intent, question, context, utterance_index, dialogue_id, answer_ids, start_position=0, end_position=0, answer=None, input_info=None, name='', *args,
                 **kwargs):
        super().__init__(input_info, *args, **kwargs)
        self.slot: str = slot
        self.intent: Union[str, list[str]] = intent
        self.utterance_index: int = utterance_index
        self.dialogue_id: str = dialogue_id
        self.question: str = question
        self.context: str = context
        self.answer: Optional[list[str]] = answer
        self.answer_ids: Optional[list[int]] = answer_ids
        self.start_position: int = start_position
        self.end_position: int = end_position
        self.name: str = name

    def __str__(self):
        return f'Utterance {self.utterance_index} of dialogue {self.dialogue_id}: {self.question} {self.context}; {self.answer} ({self.slot if self.slot else self.intent})'

    def __repr__(self):
        return str(self)


class QAInputInfo(InputInfo):
    def __init__(self, question, context, start_position=0, end_position=0, answer=None, tokenized_answer=None, input_info=None, name='', *args,
                 **kwargs):
        super().__init__(input_info, *args, **kwargs)
        self.question: str = question
        self.context: str = context
        self.answer: Optional[list[str]] = answer
        self.tokenized_answer: Optional[list[int]] = tokenized_answer
        self.start_position: int = start_position
        self.end_position: int = end_position
        self.name: str = name

        self.slot = ''
        self.intent = ''
        self.utterance_index: int = -1
        self.dialogue_id: str = ''

    def __str__(self):
        return f'Question: {self.question}; Context: {self.context}; Answer: {self.answer}'

    def __repr__(self):
        return str(self)

def find_span(tokens, attribute_value, end_token=None, start=-1) -> Tuple[int, int]:
    current_att_value_index: int = 0
    potential_start, potential_end, valid_potential_start, valid_potential_end = (
        -1,
        -1,
        -1,
        -1,
    )
    for token_i, token in enumerate(tokens):
        if token_i < start:
            continue
        if token != attribute_value[current_att_value_index]:
            potential_start, potential_end = -1, -1
            current_att_value_index = 0
        if token == attribute_value[current_att_value_index]:
            if current_att_value_index == 0:
                potential_start = token_i
            current_att_value_index += 1
            if current_att_value_index == len(
                    attribute_value
            ):  # len because we increase the value beforehand
                valid_potential_start, valid_potential_end, potential_end = (
                    potential_start,
                    token_i,
                    token_i,
                )
                if end_token is None:
                    return potential_start, potential_end
                current_att_value_index = 0
        if token == end_token:
            return valid_potential_start, valid_potential_end
    return valid_potential_start, valid_potential_end  # always -1, -1?
