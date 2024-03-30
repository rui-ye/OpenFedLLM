import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
import sys
import argparse
# sys.path.append('./')  
from peft import (
    LoraConfig, TaskType,
    PeftModel,
    get_peft_model, set_peft_model_state_dict,
    prepare_model_for_int8_training,
)

from transformers import (
    LlamaTokenizerFast, LlamaTokenizer, LlamaForCausalLM,
    LlamaConfig, LlamaForSequenceClassification,
    AutoTokenizer,AutoModelForCausalLM,
)
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from datasets import load_dataset
import csv

# relative import
from fpb import test_fpb
from fiqa import test_fiqa , add_instructions
from tfns import test_tfns
from nwgi import test_nwgi

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='', type=str)
    parser.add_argument("--peft_model", default=None, type=str)
    parser.add_argument("--use_vllm", action='store_true', default=False)
    parser.add_argument("--load_8bit", action='store_true', default=False)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=512, type=int)

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Llama has no pad token by default
    tokenizer.padding_side = 'left'
    try:
        model = LlamaForCausalLM.from_pretrained(
                args.base_model,
                device_map="auto",
                load_in_8bit=args.load_8bit,
                trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map="auto",
                load_in_8bit=args.load_8bit,
                trust_remote_code=True)
    
   
    # get peft model
    if args.peft_model:
        model = PeftModel.from_pretrained(model, args.peft_model)

    model = model.eval()
    batch_size = args.batch_size
    
    
    # FPB 1055
    instructions, acc_avg, f1_list_fpb = test_fpb(model, tokenizer, batch_size=batch_size)
    
    # FiQA 275
    instructions, acc_avg, f1_list_fiqa = test_fiqa(model, tokenizer, prompt_fun=add_instructions,
                                                    batch_size=batch_size)
   
    # tfns 2388
    instructions, acc_avg, f1_list_tfns = test_tfns(model, tokenizer, batch_size=batch_size)
    
    # NWGI 4048
    instructions, acc_avg, f1_list_nwgi = test_nwgi(model, tokenizer, batch_size=batch_size)

    


