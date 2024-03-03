import os

import warnings
import torch
from transformers import AutoTokenizer
import argparse

from bigdl.llm.transformers.qlora import PeftModel, LoraConfig
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.low_bit_linear import get_block_size
import tempfile
import shutil



current_dir = os.path.dirname(os.path.realpath(__file__))
import sys
"""
sys.path.append(common_util_path)
common_util_path = os.path.join(current_dir, '..', '..')
from common.utils import merge_adapter
"""


def merge_adapter(base_model, tokenizer, adapter_path, output_path):
    """Merge the adapter into the original model and save"""
    lora_config = LoraConfig.from_json_file(os.path.join(adapter_path, "adapter_config.json"))
    training_mode = lora_config.get("training_mode", "qlora")
    qa_lora = training_mode == "qalora"

    temp_dir = None
    if qa_lora:
        # Convert the qa-lora adapter to the correct shapes
        # The default 4-bit format for qa_lora is sym_int4
        block_size = get_block_size("sym_int4")
        temp_dir = tempfile.TemporaryDirectory()
        tmpdirname = os.path.join(temp_dir.name, "adapter")
        try:
            shutil.copytree(adapter_path, tmpdirname)
        except Exception as e:
            print(f"Failed to copy adapter dir, error: {e}")
        mid_lora_path = os.path.join(tmpdirname, "adapter_model.bin")
        adapter_path = os.path.join(adapter_path, "adapter_model.bin")

        lora = torch.load(adapter_path, map_location='cpu')
        # Get lora_a names
        tmp_keys = [key for key in lora.keys() if 'lora_A' in key]

        for tmp_key in tmp_keys:
            lora_a = lora[tmp_key] / block_size
            lora[tmp_key] = torch.repeat_interleave(lora_a, block_size, dim=1)

        torch.save(lora, mid_lora_path)
        adapter_path = tmpdirname

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_low_bit="nf4", # should load the orignal model
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )

        lora_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map={"": "cpu"},
            torch_dtype=torch.float16,
        )

        # merge weights - new merging method from peft
        lora_model = lora_model.merge_and_unload()
        lora_model.train(False)

        lora_model_sd = lora_model.state_dict()
        deloreanized_sd = {
            k.replace("base_model.model.", ""): v
            for k, v in lora_model_sd.items()
            if "lora" not in k
        }

        base_model.save_pretrained(output_path, state_dict=deloreanized_sd)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Failed to merge the adapter, error: {e}.")
    finally:
        if qa_lora and temp_dir:
           temp_dir.cleanup()
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge the adapter into the original model for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--adapter_path', type=str,)
    parser.add_argument('--output_path', type=str,)

    args = parser.parse_args()
    base_model = model_path = args.repo_id_or_model_path
    adapter_path = args.adapter_path
    output_path = args.output_path

    print("\n\nModel: ", base_model)
    print("Adapter Path: ", adapter_path)
    print("Output Path: ", output_path)
    
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    merge_adapter(base_model, tokenizer, adapter_path, output_path)
    print(f'Finish to merge the adapter into the original model and you could find the merged model in {output_path}.')
    
    