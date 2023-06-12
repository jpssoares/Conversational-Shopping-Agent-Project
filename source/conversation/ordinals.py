import spacy
import numerizer
import re

nlp = spacy.load("en_core_web_sm")
ordinal_suffixes = ["st", "nd", "rd", "th"]


def get_position(input_msg: str) -> int:
    numericals: dict[str, str] = nlp(input_msg)._.numerize()
    ordinals = [
        _create_index_from_ordinal(num)
        for numerical in numericals.values()
        for num in numerical.split()
        if _is_ordinal(num)
    ]
    return ordinals[0] if ordinals else None


def _create_index_from_ordinal(num: str) -> int:
    return int(re.sub(r"\D", "", num)) - 1


def _is_ordinal(numerical: str):
    for suffix in ordinal_suffixes:
        if suffix in numerical:
            return True
    return False
