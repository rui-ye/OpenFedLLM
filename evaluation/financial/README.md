# FinGPT: Open-Source Financial Large Language Models

## Setup
```
conda create -n fingpt python=3.8
conda activate fingpt
pip install -r requirements.txt
```

## Usage
### Dataset
* If you attend to evaluate on local datasets instead of connecting to [HuggingFace](https://huggingface.co/), **modifing the dataset_path / data_loading code** is needed.
([Example](https://github.com/rui-ye/EasyFedLLM/blob/evaluation/evaluation/financial/benchmarks/fpb.py#L43)) 
* Or you can prepare your local datasets according to [File](https://github.com/rui-ye/EasyFedLLM/blob/evaluation/evaluation/financial/data/prepare_data.ipynb)

### Run Scripts
* To reproduce our results or evaluate your model on all the datasets/tasks used in our work, you can simply run **fin_all.sh** .
```bash
bash scripts/fin_all.sh $path_to_your_model $gpu_ids
```

### Evaluation Args
```bash
CUDA_VISIBLE_DEVICES=<CUDA_DEVICES> python benchmarks/fingpt_bench.py \
  --model <MODEL_NAME> \
  --peft_model <PEFT_MODEL_PATH> \
  --batch_size <BATCH_SIZE> \
  --max_length <MAX_LENGTH> 
```

## Note
* For more information, please go to the origin repository we fork from: [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT), or you can look up the original version of [FinGPT README](https://github.com/AI4Finance-Foundation/FinGPT/blob/master/README.md). 
* [@wwh0411](https://github.com/wwh0411) simplifies the code for our usage.
