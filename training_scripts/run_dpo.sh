gpu=2
max_steps=10
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=5
sample_clients=2
lr=5e-4

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="Anthropic/hh-rlhf"
# dataset_name="HuggingFaceH4/ultrafeedback_binarized"
dataset_sample=20000
model_name_or_path="ehartford/Wizard-Vicuna-7B-Uncensored"
output_dir=./output

fed_alg="fedavg"
CUDA_VISIBLE_DEVICES=$gpu python main_dpo.py \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --learning_rate $lr \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "vicuna_v1.1" \