import hashlib
import numpy as np
from PIL import Image

def get_range(input: str, offset, range_len=4):
    offset = offset % len(input)
    return (input * 2)[offset:offset + range_len]

def get_sha256(input: str):
    hash_object = hashlib.sha256()
    hash_object.update(input.encode("utf-8"))
    return hash_object.hexdigest()

def shuffle_arr(arr, key):
    sha_key = get_sha256(key)
    key_len = len(sha_key)
    arr_len = len(arr)
    key_offset = 0
    for i in range(arr_len):
        to_index = int(get_range(sha_key, key_offset, range_len=8), 16) % (arr_len - i)
        key_offset += 1
        if key_offset >= key_len:
            key_offset = 0
        arr[i], arr[to_index] = arr[to_index], arr[i]
    return arr

def _prepare_image(image: Image.Image):
    if image.mode not in ("RGB", "RGBA"):
        if "A" in image.getbands():
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    return image

def _encrypt_array(pixel_array: np.ndarray, psw: str):
    height, width = pixel_array.shape[:2]
    x_arr = [i for i in range(width)]
    shuffle_arr(x_arr, psw)
    y_arr = [i for i in range(height)]
    shuffle_arr(y_arr, get_sha256(psw))

    for y in range(height):
        _y = y_arr[y]
        temp = pixel_array[y].copy()
        pixel_array[y] = pixel_array[_y]
        pixel_array[_y] = temp

    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    for x in range(width):
        _x = x_arr[x]
        temp = pixel_array[x].copy()
        pixel_array[x] = pixel_array[_x]
        pixel_array[_x] = temp

    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))
    return pixel_array

def _decrypt_array(pixel_array: np.ndarray, psw: str):
    height, width = pixel_array.shape[:2]
    x_arr = [i for i in range(width)]
    shuffle_arr(x_arr, psw)
    y_arr = [i for i in range(height)]
    shuffle_arr(y_arr, get_sha256(psw))

    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    for x in range(width - 1, -1, -1):
        _x = x_arr[x]
        temp = pixel_array[x].copy()
        pixel_array[x] = pixel_array[_x]
        pixel_array[_x] = temp

    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    for y in range(height - 1, -1, -1):
        _y = y_arr[y]
        temp = pixel_array[y].copy()
        pixel_array[y] = pixel_array[_y]
        pixel_array[_y] = temp

    return pixel_array

def encrypt_image_v2(image: Image.Image, psw: str):
    image = _prepare_image(image).copy()
    pixel_array = np.array(image)
    pixel_array = _encrypt_array(pixel_array, psw)
    return Image.fromarray(pixel_array, mode=image.mode)

def dencrypt_image_v2(image: Image.Image, psw: str):
    image = _prepare_image(image).copy()
    pixel_array = np.array(image)
    pixel_array = _decrypt_array(pixel_array, psw)
    return Image.fromarray(pixel_array, mode=image.mode)

decrypt_image_v2 = dencrypt_image_v2
