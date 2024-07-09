import copy
import os
os.environ['HF_HOME'] = '/lcrc/project/NEXTGENOPT/yijiang/cache'
from tqdm import tqdm
import numpy as np

import torch
from transformers import OPTForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
import wandb

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

################

"""
python main_sft_opt.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name gbharti/finance-alpaca \
    --dataset_sample 20000 \
    --fed_alg fedavg \
    --num_clients 5 \
    --sample_clients 5 \
    --max_steps 10 \
    --num_rounds 100 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --seq_length 512 \
    --peft_lora_r 64 \
    --peft_lora_alpha 64 \
    --use_peft True \
    --output_dir ./output \
    --template alpaca
"""

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# Initialize wandb with your project and optionally your API key
wandb.init(project='main_sft_opt', config=script_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

# ===== Get model =====
model = OPTForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    torch_dtype='auto',
)

# ===== Load quantized LLM if specified =====
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

# ===== Apply parameter-efficient fine-tuning =====
if script_args.use_peft:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    print("############### PEFT is not used. ###################")

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
if script_args.use_peft:
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
else:
    global_dict = model.state_dict()
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            # wandb.log({f'Training Loss Client {client}': training_loss[client][-1]})
            continue

        if script_args.use_peft:
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        else:
            model.load_state_dict(global_dict)

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        wandb.log({f'Training Loss Client {client}': training_loss[client][-1]})

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        if script_args.use_peft:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
        else:
            local_dict_list[client] = copy.deepcopy(model.state_dict())
            
    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    if script_args.use_peft:
        set_peft_model_state_dict(model, global_dict)   # Update global model
    else:
        model.load_state_dict(global_dict)

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

        # Log the saved model to wandb
        wandb.save(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

# Finalize wandb at the end of the script
wandb.finish()