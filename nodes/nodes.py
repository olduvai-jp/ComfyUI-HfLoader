import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import folder_paths
from huggingface_hub import hf_hub_download

class LoraLoaderFromHF:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "repo_name": ("STRING", {"multiline": False, "default": "lora_name"}),
                              "filename": ("STRING", {"multiline": False, "default": "lora_name"}),
                              "api_token": ("STRING", {"multiline": False, "default": ""}),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora_from_hf"

    CATEGORY = "HF_loaders"

    def load_lora_from_hf(self, model, clip, repo_name, filename, api_token, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == repo_name and self.loaded_lora[1] == filename:
                lora = self.loaded_lora[2]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            
            # Load from HF
            token = api_token if api_token != "" else None
            cache_dirs = folder_paths.get_folder_paths(Folders.HF_CACHE_DIR)
            lora_path = hf_hub_download(
                repo_name, 
                filename, 
                token=token, 
                cache_dir=cache_dirs[0]
            )
            print(f"Loaded Lora from {lora_path}")

            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (repo_name, filename, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

from typing import Callable, Union
from collections.abc import Iterable
from pathlib import Path

class Folders:
    HF_CACHE_DIR = "hf_cache_dir"

def add_extension_to_folder_path(folder_name: str, extensions: Union[str, list[str]]):
    if folder_name in folder_paths.folder_names_and_paths:
        if isinstance(extensions, str):
            folder_paths.folder_names_and_paths[folder_name][1].add(extensions)
        elif isinstance(extensions, Iterable):
            for ext in extensions:
                folder_paths.folder_names_and_paths[folder_name][1].add(ext) 


def try_mkdir(full_path: str):
    try:
        Path(full_path).mkdir()
    except Exception:
        pass

folder_paths.add_model_folder_path(Folders.HF_CACHE_DIR, str(Path(folder_paths.models_dir) / Folders.HF_CACHE_DIR))
add_extension_to_folder_path(Folders.HF_CACHE_DIR, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.HF_CACHE_DIR))