MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8
# GPUS_PER_NODE=4
# MASTER_ADDR=localhost
# MASTER_PORT=6012
# NNODES=1
# NODE_RANK=0

REPO_DIR=$(pwd)
OUT_DIR=${REPO_DIR}
VISION_TOWER_CKPT=${OUT_DIR}/work_dir/path_to_evotok_siglip256/last.pt


cd $REPO_DIR/mllm

PRETRAIN_OUT_PATH=${OUT_DIR}/work_dir/EvoTok-siglip2_256-llava-qwen2.5-7B_gen_pre_llava558k

# data path
DATA_PATH=${REPO_DIR}/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_generation_debug.json
IMG_DIR=${REPO_DIR}/data/LLaVA-Pretrain/images


LLM_PATH=Qwen/Qwen2.5-7B-Instruct

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")

export PYTHONPATH=.
export WANDB_MODE=offline

export VQGAN_DEPTH=4
export VQKD_DEPTH=16
export MM_MODE="generation"


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
    --image_aspect_ratio square \
    --mm_vision_vq_model_type siglip_256 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_with_depth_transformer True \
    --tune_mm_depth_transformer True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $PRETRAIN_OUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --run_name ${PRETRAIN_TASK_NAME} \
            2>&1 | tee ${OUT_DIR}/work_dir/logs/${PRETRAIN_TASK_NAME}_$(date +"%Y-%m-%d_%H-%M-%S").log

