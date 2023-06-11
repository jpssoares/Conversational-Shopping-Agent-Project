from models.model_utils import get_model
from transformers import AutoTokenizer
import source.config as config
import transformers
from source.conversation.predefined_messages import GET_ELEM_PROMPT

CHAT_INTENT_KEYS = [
    "user_neutral_are_you_a_bot",
    "user_neutral_what_is_your_name",
    "user_neutral_where_are_you_from",
    "user_neutral_who_do_you_work_for",
    "user_neutral_who_made_you",
]

QA_INTENT_KEYS = [
    "user_qa_check_information",
    "user_qa_product_composition",
    "user_qa_product_description",
    "user_qa_product_information",
    "user_qa_product_measurement",
]


# Set the transformer verbosity to hide the annoying warnings
transformers.logging.set_verbosity_error()

checkpoint_name = "bert-base-uncased"
config.model_name = "bertdsti"
config.start_by_loading = True
config.max_len = 128
config.load_path = "trained-models/bert-dsti-ff-new.ptbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, truncation_side="left")
model, input_function, dataloading_function = get_model(
    checkpoint_name, tokenizer, None
)


def add_special_tokens_to_model_and_tokenizer(
    model, tokenizer, special_tokens, embeddings
):
    if model is None or hasattr(model, "decoder"):
        if model is None:
            for special_token in special_tokens:
                tokenizer.add_tokens(special_token)
        return


add_special_tokens_to_model_and_tokenizer(
    None,
    tokenizer,
    [" Dontcare", "[sys]", "[usr]", "[intent]"],
    ["I don't care", "[SEP]", "[SEP]", "[CLS]"],
)


def interpreter(msg: str):
    global model
    o = input_function(tokenizer=tokenizer, question=msg)
    tokens = tokenizer.convert_ids_to_tokens(o["input_ids"][0])
    output = model.to("cpu").get_human_readable_output(o, tokens)
    intent: str = output.get_intent()
    dict_keys = output.value.keys()

    slots = []
    values = []

    for key in dict_keys:
        value = output.get_slot_value_from_key(key)
        slots.append(key)
        values.append(value)

    return intent, slots, values
