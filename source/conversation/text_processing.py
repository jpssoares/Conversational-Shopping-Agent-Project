from collections import OrderedDict

import spacy
import numerizer  # it's necessary for .numerize() to work
import re
from fuzzywuzzy import fuzz
from typing import Union

nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words |= {"want", "would", "like", "looking", "for", "searching"}
special_tokens = ["[intent]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]
ORDINA_PATTERN = r"\d+\w{2}"


def preprocess_input_msg(text: str, values: list[str]) -> str:
    unique_tokens = []
    token_dict = OrderedDict()

    for token in nlp(text):
        if not token.is_stop and token.text not in token_dict:
            token_dict[token.text] = None

    for value in values:
        for token in value.split():
            if token not in token_dict and token not in special_tokens:
                token_dict[token] = None

    unique_tokens = list(token_dict.keys())
    processed_msg = " ".join(unique_tokens)
    return processed_msg


def get_position(input_msg: str) -> Union[int, None]:
    numericals: dict[str, str] = nlp(input_msg)._.numerize()

    ordinals = [
        _create_index_from_ordinal(num)
        for numerical in numericals.values()
        for num in numerical.split()
        if _is_ordinal(num)
    ]
    ordinals.extend(
        [-1 for word in input_msg.split() if fuzz.ratio(word.lower(), "last") > 71]
    )
    ordinals.extend(
        [-1 for word in input_msg.split() if fuzz.ratio(word.lower(), "ultimate") > 71]
    )

    return ordinals[0] if ordinals else None


def _create_index_from_ordinal(num: str) -> int:
    return int(re.sub(r"\D", "", num)) - 1


def _is_ordinal(numerical: str):
    for suffix in ORDINAL_SUFFIXES:
        if suffix in numerical and re.search(ORDINA_PATTERN, numerical):
            return True
    return False
