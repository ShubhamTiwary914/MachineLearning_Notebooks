"""
    Fine Tune on Intel Xeon CPUs using Qlora, Transformers & PEFT
"""

import torch
import os
import warnings

import transformers
from transformers import LlamaTokenizer, AutoTokenizer

from transformers import BitsAndBytesConfig
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_dataset, Dataset
import argparse
from bigdl.llm.utils.isa_checker import ISAChecker


def generatorPrompt(row):
    row['prediction'] = str(row['label'])
    return row


def filterColumn(dataset, colName):
    newData = {}
    keys = dataset[0].keys()
    for key in keys:
        if(key != colName):
            newData[key] = dataset[key]
    return Dataset.from_dict(newData)
    
    

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default="Abirate/english_quotes")
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = load_dataset(dataset_path)

    data['train'] = data['train'].map(generatorPrompt)
    data['test'] = data['test'].map(generatorPrompt)

    print("\nDataset After Mapping Prediction: ")
    print(data, "\n\n")
    
    # use the max_length to reduce memory usage, should be adjusted by different datasets
    data['train'] = data['train'].map(lambda samples: tokenizer(samples["prediction"], max_length=512), batched=True)
    data['test'] = data['test'].map(lambda samples: tokenizer(samples["prediction"], max_length=512), batched=True)


    data['train'] = filterColumn(data['train'], 'label')
    data['test'] = filterColumn(data['test'], 'label')

    
    print("Dataset After Tokenisation  on Prediction: ")
    print(data, "\n\n")
    

    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="int4",  # nf4 not supported on cpu yet
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config, )


    model = model.to('cpu')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    # To avoid only one core is used on client CPU
    isa_checker = ISAChecker()
    bf16_flag = isa_checker.check_avx512()
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset= data["train"],
        eval_dataset= data["test"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=20,
            max_steps=200,
            learning_rate=2e-4,
            save_steps=100,
            bf16=bf16_flag,
            logging_steps=20,
            output_dir="./outputs",
            optim="adamw_hf",  
        ),
        # Inputs are dynamically padded to the maximum length of a batch
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
    trainer.save_model("./trained/")
    

    