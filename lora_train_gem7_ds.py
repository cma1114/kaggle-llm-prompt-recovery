###  Load base model and prep for lora fine tuning   ###
import torch
import gc
import pandas as pd
import time
import os
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import wandb
import logging
logging.basicConfig(filename='training_debug.log', level=logging.DEBUG)

hf_access_token = os.getenv('HF_TOKEN')
wandb_key = os.getenv('WANDB_API_KEY')

wandb.login(key=wandb_key)

model=None
tokenizer=None
gc.collect()
torch.cuda.empty_cache()

modelName = "google/gemma-7b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],#, "gate_proj", "up_proj", "down_proj"], #less prone to overfitting w/o mlp modules
    task_type="CAUSAL_LM",
    lora_alpha=32,#should be twice r according to lightning ai (hearsay)
    lora_dropout=0.1,
    bias="none",
)

tokenizer = AutoTokenizer.from_pretrained(modelName)
from accelerate import PartialState
device_string = PartialState().process_index
print("device_string",device_string)
model = AutoModelForCausalLM.from_pretrained(modelName, quantization_config=bnb_config, token=hf_access_token, device_map={'':device_string})
# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, "./output/checkpoint-600",is_trainable=True)
######model = model.merge_and_unload()

model.train()
###model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
###################################################


###      Read in and format data to train on    ###
import pandas as pd
from datasets import Dataset
dsname = "./data/mixed_dataset_big"
df = pd.read_csv(dsname+"_train.csv")
df = df.sample(frac=1).reset_index(drop=True)
data = Dataset.from_pandas(df)

def format_prompt(ds):
    template = ("The following `Original Text` passage has been rewritten into `Rewritten Text` by the `Gemma 7b-it` "
            "LLM with a certain prompt. Your task is to carefully analyze the differences between the `Original Text` and `Rewritten Text`, "
            "and try to infer the specific prompt that was likely given to the LLM to rewrite the text in this way. Consider "
            "the writing style, meter, tone, etc of the rewritten text, and think about how it differs from the original. Then respond ONLY with "
            "the prompt that you predict would have yielded that change. Focus on the DIFFERENCE between the original and rewritten versions, not what is similar."
            f"\n\nOriginal Text:\n{{original_text}}\n\nRewritten Text:\n{{rewritten_text}}\n\nPredicted Prompt:\n{{rewrite_prompt}}")
    template = ("<bos><start_of_turn>user\nThe following `Original Text` passage has been rewritten into `Rewritten Text` by the `Gemma 7b-it` "
            "LLM with a certain prompt. Your task is to carefully analyze the differences between the `Original Text` and `Rewritten Text`, "
            "and try to infer the specific prompt that was likely given to the LLM to rewrite the text in this way. Consider "
            "the writing style, meter, tone, etc of the rewritten text, and think about how it differs from the original. Then respond ONLY with "
            "the prompt that you predict would have yielded that change. Focus on the DIFFERENCE between the original and rewritten versions, not what is similar."
            f"\n\nOriginal Text:\n{{original_text}}\n\nRewritten Text:\n{{rewritten_text}}\n\nPredicted Prompt:\n<end_of_turn><start_of_turn>model\n{{rewrite_prompt}}<end_of_turn><eos>")
    prompts = [template.format(
        original_text=ot, 
        rewritten_text=rt, 
        rewrite_prompt=rp
    ) for ot, rt, rp in zip(ds['original_text'], ds['rewritten_text'], ds['rewrite_prompt'])]
    return {'text': prompts}
prompts = data.map(format_prompt, batched=True)
prompts = prompts.remove_columns(data.column_names)
#print(prompts[0])

df = pd.read_csv(dsname+"_val.csv")
df = df.sample(frac=1).reset_index(drop=True)
data = Dataset.from_pandas(df)
prompts_val = data.map(format_prompt, batched=True)
prompts_val = prompts_val.remove_columns(data.column_names)
#print(prompts_val[0])

max_seq_length=2048
print(len(prompts))
def filter_tokens(example):
    tokens = tokenizer(example['text'])['input_ids']
    return len(tokens) < max_seq_length
prompts = prompts.filter(filter_tokens)
print(len(prompts))
#print(prompts[0])

#print(len(prompts_val))
def filter_tokens(example):
    tokens = tokenizer(example['text'])['input_ids']
    return len(tokens) < max_seq_length
prompts_val = prompts_val.filter(filter_tokens)
print(len(prompts_val))
#print(prompts_val[0]['text'])
######################################################


###        Fine tune         ###
print("Begin Fine tuning section")
import transformers
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
gc.collect()
torch.cuda.empty_cache()

if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side = 'right'
response_template = "<start_of_turn>model"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

class InferenceCallback(TrainerCallback):
    def __init__(self, eval_dataset, step_interval=10):
        self.eval_dataset = eval_dataset
        self.step_interval = step_interval

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Run inference every `step_interval` steps
        if state.global_step % self.step_interval == 0 and state.global_step > 0:
            # Pick a single example to run inference (change the index as needed)
            example = self.eval_dataset[0]['text'].split("<start_of_turn>model")[0] + "<start_of_turn>model"
#            example = "<start_of_turn>user\n" + example + "<end_of_turn>\n<start_of_turn>model "
            model = kwargs['model']
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")
                outputs = model.generate(**inputs,max_new_tokens=60,use_cache=True)#, penalty_alpha=0.6, num_beams=2)
                rewrite_prompt = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                #print("Input:", example)
                print(f"Inference output at step {state.global_step}: {rewrite_prompt}")
            model.train()  # Set the model back to train mode

# hyperparameters
batch_size = 1 #runs out of memory with 2
learning_rate=2e-5#2e-4#3e-4 slower leads to less overfitting
num_train_epochs=4
weight_decay=0.0
gradient_accumulation_steps=4

deepspeed_config = {
    "train_micro_batch_size_per_gpu": batch_size,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "round_robin_gradients": True
    }
}

training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=20,
        warmup_steps=10,
        save_steps = 100,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir="outputs_mxd_big_cplt_epoch2",
        report_to='none',#wandb',
        optim="paged_adamw_8bit"
        ,deepspeed=deepspeed_config#"path/to/deepspeed_config.json"
        ,resume_from_checkpoint='./output/checkpoint-600'
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = prompts,
    eval_dataset=prompts_val.select(range(100)),
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, dataset_num_proc = 2,#True, 
    args=training_arguments,
    peft_config=lora_config,
    data_collator=collator,
    callbacks=[InferenceCallback(eval_dataset=prompts_val, step_interval=20)]
)
print("Begin Fine tuning")
trainer.train()
print("Done Fine tuning")
model.push_to_hub("cackerman/rewrites_gemma7_ft_mxdds_big", token = hf_access_token)
######################################################
