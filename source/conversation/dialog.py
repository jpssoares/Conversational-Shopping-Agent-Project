from models.model_utils import get_model
from transformers import AutoTokenizer
import source.config as config
import transformers

chat_intent_keys = [
 "user_neutral_are_you_a_bot",
 "user_neutral_do_you_have_pets",
 "user_neutral_fun_fact",
 "user_neutral_how_old_are_you",
 "user_neutral_meaning_of_life",
 "user_neutral_tell_joke",
 "user_neutral_what_are_your_hobbies",
 "user_neutral_what_is_your_name",
 "user_neutral_where_are_you_from",
 "user_neutral_who_do_you_work_for",
 "user_neutral_who_made_you"
]

qa_intent_keys = [
 'user_qa_check_information',
 'user_qa_product_composition',
 'user_qa_product_description',
 'user_qa_product_information',
 'user_qa_product_measurement'
 ]

get_elem_prompt = "I am building a dialog state tracking machine, and my model has a slot_key named \'element\'. \'element\' represent the position of the element in a given sequence. For example, \'what is the brand of the third product?\' will give me a value for \'element\'  that is 3. If you cant find a value for \'element\', please set is as unknown. If the user is refering to more than one position, set \'element\' as all.\nWhat would be the key-value pair for this phrase:\n\'{input}\'\nPlease return the result inseide curly brackets."


# Set the transformer verbosity to hide the annoying warnings
transformers.logging.set_verbosity_error()

# load model and tokenizer
checkpoint_name = 'bert-base-uncased'
config.model_name = 'bertdsti'
config.start_by_loading = True
config.max_len = 128
config.load_path = 'trained-models/bert-dsti-ff-new.ptbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, truncation_side='left')
model, input_function, dataloading_function = get_model(checkpoint_name, tokenizer, None)

def add_special_tokens_to_model_and_tokenizer(model, tokenizer, special_tokens, embeddings):
    # TODO instead of checking for the shared param you should really just have a good way to tell whether the model has some sort of decoder
    if model is None or hasattr(model, 'shared'):
        if model is None:
            for special_token in special_tokens:
                tokenizer.add_tokens(special_token)
        return
    
add_special_tokens_to_model_and_tokenizer(
    None,
    tokenizer,
    [' Dontcare', '[sys]', '[usr]', '[intent]'],
    ['I don\'t care', '[SEP]', '[SEP]', '[CLS]']
)

def interpreter(msg):
    o = input_function(tokenizer=tokenizer, question=msg)
    tokens = tokenizer.convert_ids_to_tokens(o["input_ids"][0])
    output = model.get_human_readable_output(o, tokens)
    intent = output.get_intent()
    keys = output.value.keys()
    values = []

    for key in keys:
        value = output.get_slot_value_from_key(key)
        values.append(value)

    return intent, keys, values