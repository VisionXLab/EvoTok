data_dir=data/llava15

hf download --repo-type dataset liuhaotian/LLaVA-Pretrain  --cache-dir $data_dir

save_dir=$data_dir/LLaVA-Pretrain
mkdir -p $save_dir
unzip $data_dir/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/images.zip \
    -d $save_dir/images

cp $data_dir/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/blip_laion_cc_sbu_558k_meta.json $save_dir
cp $data_dir/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/blip_laion_cc_sbu_558k.json $save_dir
