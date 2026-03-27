import json
import os
import re
import time

import folder_paths
import numpy as np
import torch
from PIL import Image, PngImagePlugin

from .core.core import decrypt_image_v2, encrypt_image_v2

def _sanitize_name(value: str, fallback: str = "Encrypted"):
    value = (value or "").strip()
    value = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")
    return value or fallback

def _tensor_to_pil(image_tensor):
    array = image_tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim != 3:
        raise ValueError("Expected a single IMAGE tensor with shape [H, W, C].")
    return Image.fromarray(array)

def _pil_to_tensor(image: Image.Image):
    image = image.convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)

def _metadata_to_pnginfo(prompt=None, extra_pnginfo=None, additional=None):
    pnginfo = PngImagePlugin.PngInfo()

    def add_value(key, value):
        if value is None:
            return
        if isinstance(value, bytes):
            text = value.decode("utf-8", "replace")
        elif isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
        pnginfo.add_text(str(key), text)

    if prompt is not None:
        add_value("prompt", prompt)

    if isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            add_value(key, value)

    if isinstance(additional, dict):
        for key, value in additional.items():
            add_value(key, value)

    return pnginfo

def _unique_path(folder, prefix, index):
    safe_prefix = _sanitize_name(prefix, "Encrypted")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{safe_prefix}_{stamp}_{index:05d}"
    filename = f"{base}.png"
    path = os.path.join(folder, filename)
    counter = 1
    while os.path.exists(path):
        filename = f"{base}_{counter}.png"
        path = os.path.join(folder, filename)
        counter += 1
    return filename, path

class EncryptAndSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "password": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "Encrypted"}),
                "subfolder": ("STRING", {"default": "encrypted"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "saved_paths")
    FUNCTION = "execute"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def execute(self, images, password, filename_prefix="Encrypted", subfolder="encrypted", prompt=None, extra_pnginfo=None):
        output_root = folder_paths.get_output_directory()
        target_folder = os.path.join(output_root, _sanitize_name(subfolder, "encrypted"))
        os.makedirs(target_folder, exist_ok=True)

        saved_paths = []
        preview_images = []

        for idx in range(images.shape[0]):
            pil_image = _tensor_to_pil(images[idx])
            encrypted = encrypt_image_v2(pil_image, password)
            filename, path = _unique_path(target_folder, filename_prefix, idx)
            pnginfo = _metadata_to_pnginfo(prompt=prompt, extra_pnginfo=extra_pnginfo)
            encrypted.save(path, format="PNG", pnginfo=pnginfo)
            saved_paths.append(path)
            preview_images.append(_pil_to_tensor(pil_image))

        preview_batch = torch.cat(preview_images, dim=0) if preview_images else images
        return (preview_batch, "\n".join(saved_paths))

class DecryptImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "password": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "image"

    def execute(self, images, password):
        output = []
        for idx in range(images.shape[0]):
            pil_image = _tensor_to_pil(images[idx])
            decrypted = decrypt_image_v2(pil_image, password)
            output.append(_pil_to_tensor(decrypted))
        return (torch.cat(output, dim=0) if output else images,)

NODE_CLASS_MAPPINGS = {
    "EncryptAndSaveImage": EncryptAndSaveImage,
    "DecryptImage": DecryptImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EncryptAndSaveImage": "Encrypt And Save Image",
    "DecryptImage": "Decrypt Image",
}
