import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import sys

base_model = "NousResearch/Llama-2-7b-chat-hf"

def create_data():

    train_dataset = load_dataset('json', data_files='llama_file.json', split='train')
    dataset = Dataset.from_dict({'text':train_dataset['text'][0]})

    return dataset


def load_model():
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model

def load_tokeizer():
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def run_trainer(new_path, epochs=1):

    dataset = create_data()
    model = load_model()
    tokenizer = load_tokeizer()
    print("run_trianer")
    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_params = TrainingArguments(
        output_dir="/results",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_args,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    # Train model
    trainer.train()
    trainer.model.save_pretrained(new_path)
    print("Saved model at ",new_path )

if __name__ == '__main__':
    #new_model = "llama-2-7b-trained"
    new_model = sys.argv[1]
    if not os.path.exists(new_model):
        os.mkdir(new_model)
    run_trainer(new_model)




