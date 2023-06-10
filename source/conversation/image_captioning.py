import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import source.conversation.gpt as gpt
from typing import Union, Any
from ..controller import decode_img

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-large"
# )

get_clothing_items_prompt = "Please return a string array with the different clothes, based on this input:\n'{input}'\nPlease only include the array in your response. Use \" instead of ' in your response."


def get_clothing_items_from_caption(input_msg):
    # get str from gpt
    txt = gpt.get_gpt_answer(get_clothing_items_prompt.format(input=input_msg))
    print(txt)

    # parse string
    words = []
    x = txt.find('"') + 1
    while x != 0:
        x = txt.find('"') + 1
        y = txt[x:].find('"')
        words.append(txt[x : x + y])
        txt = txt[x + y + 1 :]
    words.remove("")
    return words


# Aux function that takes a list and an element of the list and return the same list with the elem in the first position
# and the initial position of the element
def array_put_elem_first(arr, elem):
    if len(arr) == 1 or len(arr) == 0:
        return -1, arr
    new_arr = [elem]
    idx = arr.index(elem)
    arr.remove(elem)
    return idx, new_arr + arr


def get_matching_clothes_quey(clothes, keys, values):
    try:
        # prioritize category
        idx, keys = array_put_elem_first(keys, "category")
        if idx != -1:
            _, values = array_put_elem_first(values, values[idx])

        # return first partial match in clothes array
        for value in values:
            for item in clothes:
                if value in item:
                    return item
    except:
        return None


def get_caption_for_image(input: Union[str, Any]):
    try:
        if type(input) is str:
            raw_image = Image.open(requests.get(input, stream=True).raw).convert("RGB")
        else:
            raw_image = decode_img(input)

        # conditional image captioning
        text = "a person wearing"
        inputs = processor(raw_image, text, return_tensors="pt")

        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True)
    except:
        return None
    return result
