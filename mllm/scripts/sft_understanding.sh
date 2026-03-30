MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8
# GPUS_PER_NODE=8
# MASTER_ADDR=localhost
# MASTER_PORT=6012
# NNODES=1
# NODE_RANK=0


REPO_DIR=$(pwd)
OUT_DIR=$(pwd)

cd $REPO_DIR/mllm

PRETRAIN_OUT_PATH=${OUT_DIR}/work_dir/EvoTok-siglip2_256-llava-qwen2.5-7B_pre_llava558k
SFT_OUT_PATH=${OUT_DIR}/work_dir/EvoTok-siglip2_256-llava-qwen2.5-7B_sft_llava665k

VISION_TOWER_CKPT=${OUT_DIR}/work_dir/path_to_evotok_siglip256/last.pt

DATA_PATH=${REPO_DIR}/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json
IMG_DIR=${REPO_DIR}/data/finetune

LLM_PATH=Qwen/Qwen2.5-7B-Instruct

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")
SFT_TASK_NAME=$(basename "${SFT_OUT_PATH%/}")

export PYTHONPATH=.
export WANDB_MODE=offline

export VQGAN_DEPTH=4 
export VQKD_DEPTH=16
export MM_MODE="understanding"




torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_PATH \
    --version qwen_2_5 \
    --data_path $DATA_PATH \
    --image_folder $IMG_DIR \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type EVOTOK \
    --pretrain_mm_mlp_adapter $PRETRAIN_OUT_PATH/mm_projector.bin \
    --mm_vision_vq_model_type siglip_256 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SFT_OUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
     --run_name ${SFT_TASK_NAME} \
                2>&1 | tee ${OUT_DIR}/work_dir/logs/${SFT_TASK_NAME}_$(date +"%Y-%m-%d_%H-%M-%S").log

