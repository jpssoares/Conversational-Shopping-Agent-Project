from typing import Tuple, Any

from data_utils.dataloading_functions import qa_specific_dataloading, slot_filling_dataloading
from models import BERTDST, BERTDSTI, model_input_functions
import source.config as config
import torch

def get_model(checkpoint_name: str, tokenizer, accel) -> Tuple[Any, Any, Any]:
    model_name = config.model_name
    if model_name == 'bertdsti':
        model, dataloading_function = BERTDSTI.BERTDSTI(checkpoint_name, tokenizer), slot_filling_dataloading
        input_function = get_simple_encoder_input_funcion(system_token='[sys]', slot_filling_to_prepend='[intent]')
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    if config.start_by_loading:
        print(f'Reloading model at {config.load_path}...')
        model = torch.load(config.load_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model, input_function, dataloading_function


def get_simple_encoder_input_funcion(to_prepend='Yes . No . Dontcare .', system_token='[sys]', slot_filling_to_prepend='',**kwargs):
    return model_input_functions.TypicalTransformerEncoderQAInputFunction(
        to_prepend=to_prepend,
        max_sequence_len=config.max_len,
        user_token='[usr]',
        system_token=system_token,
        slot_filling_to_prepend=slot_filling_to_prepend,
        **kwargs,
    )

def get_simple_encoder_input_funcion_with_more_prepend(to_prepend='Yes . No . Dontcare .', system_token='[sys]', slot_filling_to_prepend='',**kwargs):
    return model_input_functions.TypicalTransformerEncoderQAInputFunctionWithExtraSlotInfo(
        to_prepend=to_prepend,
        max_sequence_len=config.max_len,
        user_token='[usr]',
        system_token=system_token,
        slot_filling_to_prepend=slot_filling_to_prepend,
        **kwargs,
    )

def get_qanlu_simple_decoder_input_function(to_prepend=''):
    return model_input_functions.TypicalTransformerEncoderQAInputFunction(
        to_prepend=to_prepend,
        max_sequence_len=config.max_len,
        user_token='user:',
        system_token='agent:'
    )

def get_qanlu_t5_prompted_input_function(to_prepend_a='', to_prepend_b=''):
    return model_input_functions.T5PPromptedInputFunction(
        to_prepend_a=to_prepend_a,
        to_prepend_b=to_prepend_b,
        max_sequence_len=config.max_len,
        user_token='; ',
        system_token=' ',
        to_prepend=to_prepend_a + ' ' + to_prepend_b, # TODO this should be removed to facilitate more variety in prompting inputs
    )