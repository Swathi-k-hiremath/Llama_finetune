import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

base_model = "NousResearch/Llama-2-7b-chat-hf"
def load_model(new_model):

    print("loading_new_model and tokenizer")
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        new_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def evaluate_model(new_model):
    model,tokenizer = load_model(new_model=new_model)
    #print("pipeline generation")
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=200,
        repetition_penalty=1.1
    )
    return generator


def test_model(new_model,prompt):

    generator = evaluate_model(new_model=new_model)
    res = generator(prompt)
    print(res[0]["generated_text"])


if __name__ == '__main__':
    #new_model = "llama-2-7b-trained"
    sys_prompt = "[INST] <> Write an appropriate description for the given title. <> \n"
    inp = input(">>>")
    prompt = sys_prompt + inp + " [/INST]\n\n"
    model = sys.argv[1]
    test_model(model,prompt)



