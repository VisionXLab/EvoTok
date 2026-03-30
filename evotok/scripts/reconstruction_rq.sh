REPO_DIR=$(pwd)
cd $REPO_DIR

MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

# GPUS_PER_NODE=4
# MASTER_ADDR=localhost
# MASTER_PORT=6002
# NNODES=1
# NODE_RANK=0

sample_dir="${REPO_DIR}/path_to_evotok/eval_imagenet"
ckpt_path="${REPO_DIR}/path_to_evotok/last.pt"


cd evotok

export OMP_NUM_THREAD
export PYTHONPATH=.

data_path="${REPO_DIR}/data/LLaVA-Pretrain/images/00000"

torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
        tokenizer/reconstruction_rq_ddp.py \
        --vq-model EvoTok \
        --vqgan-depth 4 \
        --vqkd-depth 16 \
        --restart-unused-codes \
        --sample-dir=$sample_dir \
        --dataset "coco" \
        --data-path $data_path \
        --per-proc-batch-size 8 \
        --codebook-size=32768 \
        --codebook-embed-dim 32 \
        --image-size 256 \
        --image-size-eval 256 \
        --teacher "siglip_256" \
        --vq-ckpt=$ckpt_path &&
        

        

src_dir1=$sample_dir/samples/ &&
src_dir2=$sample_dir/gts/ &&
python3 -m evaluations.vq.pytorch_fid $src_dir1 $src_dir2 --device cuda:0 --results_path $sample_dir


set +x
target_folder_path=$sample_dir/demos/
mkdir -p "$target_folder_path"
files=()
for file in $(ls "$src_dir1"); do
    if [[ "$file" == *.png ]]; then
        files+=("$file")
    fi
done

for ((i=0; i<5 && i<${#files[@]}; i++)); do
    selected_file="${files[$i]}"
    cp "$src_dir1/$selected_file" "$target_folder_path"
done

total_files=${#files[@]}
for ((i=total_files-5; i<total_files && i>=0; i++)); do
    selected_file="${files[$i]}"
    cp "$src_dir1/$selected_file" "$target_folder_path"
done

# rm -rf $src_dir1
# rm -rf $src_dir2

echo "Selected files have been copied to $target_folder_path"