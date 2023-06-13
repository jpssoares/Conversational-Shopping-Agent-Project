import base64
from PIL import Image
import io
import validators
import requests
from typing import Union


def load_image(input_msg: str, input_img: Union[str, None] = None):
    if input_img is None:
        input_img = get_image_from_url(input_msg)
    else:
        input_img = decode_img(input_img)
    return input_img


def decode_img(input_image_query: str):
    q_image = base64.b64decode(input_image_query.split(",")[1])
    image = Image.open(io.BytesIO(q_image))
    return image


def get_image_from_url(input: str):
    print(f"input: {input}")
    for part in input.split():
        if validators.url(part):
            return Image.open(requests.get(part, stream=True).raw).convert("RGB")
    return None
