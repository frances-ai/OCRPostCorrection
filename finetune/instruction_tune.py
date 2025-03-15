import argparse
import logging
import os

import pandas as pd
import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

def format_instruction(sample):
    return f"""### Instruction:
Fix the OCR errors in the provided text.

### Input:
{sample['ocr text']}

### Response:
{sample['ground truth']}
"""

def main(args):

    # load config from yaml file
    config_filepath = os.path.join("/mnt/ceph_rbd", args.config)
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    if config and 'use_flash_attention' in config:
        use_flash_attention = config['use_flash_attention']
    else:
        use_flash_attention = False

    # Hugging Face model id
    model_name = args.model_name

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    logging.info(f"Loading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map="auto"
    )
    model.config.pretraining_tp = 1

    logging.info(f"Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Load training dataset
    output_dir = os.path.join("/mnt/ceph_rbd", args.output_dir)
    data_filepath = os.path.join("/mnt/ceph_rbd", args.data)
    logging.info(f"Loading training data from {data_filepath}...")
    train_dataset = pd.read_json(data_filepath, orient='index')
    train_dataset = Dataset.from_pandas(train_dataset)

    # Setup SFTTrainer
    default_sft_config = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_32bit",
        "logging_steps": 10,
        "save_strategy": 'no',
        "learning_rate": 2e-4,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": 'constant',
        "disable_tqdm": False,
        "max_seq_length": 1024,  # previously in SFTTrainer
        "packing": True,  # previously in SFTTrainer
    }

    if config and "sft_config" in config:
        sft_config = config['sft_configs']
        sft_config['learning_rate'] = float(sft_config['learning_rate'])
    else:
        sft_config = default_sft_config

    logging.info("Setting up SFTTrainer...")
    train_args = SFTConfig(
        output_dir=output_dir,
        **sft_config
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_instruction,
        args=train_args,
    )

    logging.info("Start training ......")
    trainer.train()

    logging.info(f"Saving model to {output_dir} ...")
    trainer.save_model(args.output_dir)
    logging.info("Done.")


if __name__ == '__main__':
    # Parse arguments for model/config/data
    parser = argparse.ArgumentParser(description='Instruction-tuning Llama 2 for OCR PostCorrection')
    parser.add_argument('--model_name', type=str, help='Name of model', required=True)
    parser.add_argument('--data', type=str, help='Path to training dataset', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to output directory', required=True)
    parser.add_argument('--config', type=str, help='Path to config yaml file')
    args = parser.parse_args()

    main(args)







