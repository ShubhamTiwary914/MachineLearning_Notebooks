import torch
import time
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# Refer to https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML#prompt-template-llama-2-chat
LLAMA2_PROMPT_FORMAT = """
[INST] <<SYS>>
Rate this review by Negative as 0 and Positive as 1
<</SYS>>
{prompt}[/INST]
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="I hate this movie so much!",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=20,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        st = time.time()
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        output = model.generate(input_ids, max_new_tokens=args.n_predict)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)

        end = time.time()
        print(f'\nRun time: {end-st} s')