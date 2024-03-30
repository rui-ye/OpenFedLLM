import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch

dic = {
    'strong negative':"negative",
    'moderately negative':"negative",
    'mildly negative':"neutral",
    'strong positive':"positive",
    'moderately positive':"positive",
    'mildly positive':'neutral',
    'neutral':'neutral',
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
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def test_nwgi(model, tokenizer, batch_size, prompt_fun = None ):
    dataset = datasets.load_dataset("oliverwang15/news_with_gpt_instructions")
    dataset = dataset['test'].to_pandas()
    dataset['output'] = dataset['label'].apply(lambda x:dic[x])

    if prompt_fun is None:
        dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)
    dataset["input"] = dataset["news"]

    dataset = dataset[['input', 'output', 'instruction']]
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    
    total_steps = dataset.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_length=512)
        res_sentences = [tokenizer.decode(i) for i in res]
        out_text = [o.split("Response: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return dataset, acc, [f1_macro, f1_micro, f1_weighted]
