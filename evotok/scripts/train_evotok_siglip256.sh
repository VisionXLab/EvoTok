set -x

REPO_DIR=$(pwd)
cd $REPO_DIR

MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

# GPUS_PER_NODE=8
# MASTER_ADDR=localhost
# MASTER_PORT=6000
# NNODES=1
# NODE_RANK=0

tag=evotok_cc12m_imagenet_res256

mkdir -p ${REPO_DIR}/work_dir/logs

cd evotok
export OMP_NUM_THREAD
export PYTHONPATH=.

torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
        tokenizer/rq_train.py "$@" \
        --vq-model EvoTok \
        --vqgan-depth 4 \
        --vqkd-depth 16 \
        --vqkd-weight 1.0 \
        --vq-warmup 2000 \
        --vqkd-loss cosine+fold \
        --restart-unused-codes \
        --dataset "multiple" \
        --data-path "cc12m:${REPO_DIR}/data/cc12m-wds/cc12m+imagenet:${REPO_DIR}/data/ImageNet-1K/train" \
        --val-data-path "${REPO_DIR}/data/ImageNet-1K/val" \
        --teacher "siglip_256" \
        --mixed-precision bf16 \
        --image-size 256 \
        --codebook-size 32768 \
        --codebook-embed-dim 32 \
        --iterations 500000 \
        --global-batch-size 256 \
        --ckpt-every 100000 \
        --ckpt-last-every 10000 \
        --eval-every 10000 \
        --log-every 100 \
        --results-dir "${REPO_DIR}/work_dir/$tag/" \
                2>&1 | tee ${REPO_DIR}/work_dir/logs/${tag}_$(date +"%Y-%m-%d_%H-%M-%S").log