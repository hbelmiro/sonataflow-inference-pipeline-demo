import os
import json
import numpy as np
from PIL import Image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def to_kserve(image_path):
    # Load image
    im = Image.open(image_path)

    # Pad to 640x640 square
    im = expand2square(im, (114, 114, 114))

    # Convert to np array of correct shape
    arr = np.transpose(np.array(im), (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    # Normalize the image
    normalized_image_array = (arr / 255.0)

    # Write to json
    row = {"name": "images", "shape": arr.shape, "datatype": "FP32", "data": normalized_image_array.tolist()}
    json_string = json.dumps({"inputs": [row]})

    return json.loads(json_string)
