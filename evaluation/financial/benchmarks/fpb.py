import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch

dic = {
        0:"negative",
        1:'neutral',
        2:'positive',
    }

# def format_example(example: dict) -> dict:
#     context = f"Instruction: {example['instruction']}\n"
#     if example.get("input"):
#         context += f"Input: {example['input']}\n"
#     context += "Answer: "
#     target = example["output"]
#     return {"context": context, "target": target}

def format_example(example: dict) -> dict:
    context = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n {example['instruction']}\n\n"
    if example.get("input"):
        context += f"### Input:\n {example['input']}\n\n"
    # context += "Answer: "
    context += "### Response: "
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    if 'negative' in x or 'Negative' in x or 'Neg' in x:
        return 'negative'
    elif 'positive' in x or 'Positive' in x or 'Pos' in x:
        return 'positive'
    
    else:
        return 'neutral'

def test_fpb(model, tokenizer, batch_size=8, prompt_fun=None ):
    instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed = 42)['test']

    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x:dic[x])

    
    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose only one answer from {/negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis = 1)
    
    instructions[["context","target"]] = instructions.apply(format_example, axis = 1, result_type="expand")

    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")


    context = instructions['context'].tolist()
    
    total_steps = instructions.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_length=512, top_k=100, temperature=0.2)
        
            
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        
        out_text = [o.split("Response: ")[1] for o in res_sentences]
        
       
        out_text = [change_target(x) for x in out_text]
    
        out_text_list += out_text
        torch.cuda.empty_cache()
        

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    
    instructions["new_out"] = instructions["out_text"].apply(change_target)
    print(instructions['new_out'].value_counts())

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average = "macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average = "micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return instructions, acc, [f1_macro, f1_micro, f1_weighted]