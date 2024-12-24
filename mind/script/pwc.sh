set -ex

# pip install -r requirements.txt
wandb login d36709b853adbec68a95bfe40570da5188ac579e
wandb online   
wandb enabled

NUM_GPUS=8

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline_for_pwc.py \
    configs/llama3/pwc.json 2>&1 | tee output/llama3/pwc.log


# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  pipeline_for_multimodal.py \
#     configs/base/llama3/author.json 2>&1 | tee output/base/llama3/author.log
