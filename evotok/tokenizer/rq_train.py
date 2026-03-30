# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.utils as vutils

import os
import time
import argparse
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from typing import Tuple, List

import wandb
from evaluations import FIDCalculator
from utils.logger import create_logger, enable_wandb, enable_tensorboard, log_infos
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr, center_crop_arr
from dataset.build import build_dataset
from tokenizer.vq_model import VQ_models
from tokenizer.modules.vq_loss import VQLoss

# import debugpy
# debugpy.listen(address = ('0.0.0.0', 5676))
# debugpy.wait_for_client() 
# breakpoint() #在下一句代码处暂停


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        writer = enable_tensorboard(args.results_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.cloud_save_path is not None:
            time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
            cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
        else:
            cloud_checkpoint_dir = None
    else:
        logger = create_logger(None)
        writer = None
        checkpoint_dir = None
        cloud_checkpoint_dir = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    print(args.restart_unused_codes)
    print("="*100)
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        restart_unused_codes=args.restart_unused_codes,
        vqgan_depth=args.vqgan_depth,
        vqkd_depth=args.vqkd_depth,
        dropout_p=args.dropout_p,
        teacher=args.teacher,
        vq_warmup=args.vq_warmup
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    else:
        ema = None
    vq_model = vq_model.to(device)

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        teacher=args.teacher,
        vqkd_weight=args.vqkd_weight,
        vqkd_loss=args.vqkd_loss
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    fid_calculator = FIDCalculator(device=device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    # Setup optimizer
    logger.info(f"args.lr = {args.lr}")
    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Setup data:
    train_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = build_dataset(args, transform=train_transform, split="train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Train dataset contains {len(train_dataset):,} images ({args.data_path})")

    val_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    val_dataset = build_dataset(args, transform=val_transform, split="val")
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )   
    logger.info(f"Val dataset contains {len(val_dataset):,} images ({args.val_data_path})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        model_state = checkpoint["model"]
        vq_model.load_state_dict(model_state, strict=False)
        logger.info("Loaded model from checkpoint.")

        if args.ema:
            ema.load_state_dict(checkpoint["ema"])

        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            logger.info("Optimizer starting from scratch.")
        try:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        except:
            logger.info("Discriminator starting from scratch.")
        try:
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            logger.info("Discriminator optimizer starting from scratch.")
            
        if not args.finetune:
            if args.by_epoch:
                if "steps" in checkpoint:
                    train_steps = checkpoint["steps"]
                    start_epoch = int(train_steps / int(len(train_dataset) / args.global_batch_size))
                else:
                    train_epoch = int(args.vq_ckpt.split('/')[-1].split('.')[0])
                    start_epoch = train_epoch + 1
                train_steps = int(start_epoch * int(len(train_dataset) / args.global_batch_size))
            else:
                train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
                start_epoch = int(train_steps / int(len(train_dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu])
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    print(f"********* dtype: {ptdtype}. *********")
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    if args.by_epoch:
        logger.info(f"Training for {args.epochs} epochs...")
    else:
        logger.info(f"Training for {args.iterations} iterations...")
        args.epochs = 100000

    # Initialize wandb
    if rank == 0:
        enable_wandb(project="cloud-VQ", config=args,
                     exp_name=os.path.basename(os.path.dirname(args.results_dir)),
                     save_dir=args.results_dir)
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        pbar = tqdm(train_loader, desc='train', leave=True) if rank==0 else train_loader

        for mini_iter, (x, y) in enumerate(pbar):
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss = vq_model(imgs)

                loss_gen, loss_dict_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=vq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every)

            scaler.scale(loss_gen).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc, loss_dict_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                if rank == 0:
                    info = f"Train (Epoch-{epoch} Step-{train_steps:07d}) Loss: {avg_loss:.4f}"
                    pbar.set_description(desc=info, refresh=False)

                    log_infos({**loss_dict_gen, **loss_dict_disc}, train_steps, prefix="train", writer=writer, epoch=epoch)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if not args.by_epoch:
                # is_final = (epoch == args.epochs - 1) and (mini_iter == len(pbar) - 1)
                is_final = (train_steps == args.iterations)
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    save_ckpt(f"iter-{train_steps:07d}", args, vq_model, optimizer, vq_loss, optimizer_disc, train_steps, ema, checkpoint_dir, logger, cloud_checkpoint_dir, rank)

                if (train_steps % args.ckpt_last_every == 0 and train_steps > 0) or is_final:
                    save_ckpt(f"last", args, vq_model, optimizer, vq_loss, optimizer_disc, train_steps, ema, checkpoint_dir, logger, cloud_checkpoint_dir, rank)

                if (train_steps % args.eval_every == 0 and train_steps > 0) or is_final:
                    eval_outputs = evaluate_one_round(val_loader, vq_model, fid_calculator, rank, device, ptdtype, logger)
                    if rank == 0:
                        log_infos(eval_outputs, train_steps, "test", writer)

                if is_final:
                    break

        if args.by_epoch:
            is_final = (epoch == args.epochs - 1)
            if epoch % args.ckpt_every == 0 and epoch > 0:
                save_ckpt(f"epoch-{epoch:07d}", args, vq_model, optimizer, vq_loss, optimizer_disc, train_steps, ema, checkpoint_dir, logger, cloud_checkpoint_dir, rank)

            if (epoch % args.ckpt_last_every == 0 and epoch > 0) or is_final:
                save_ckpt(f"last", args, vq_model, optimizer, vq_loss, optimizer_disc, train_steps, ema, checkpoint_dir, logger, cloud_checkpoint_dir, rank)

            if (epoch % args.eval_every == 0 and epoch > 0) or is_final:
                eval_outputs = evaluate_one_round(val_loader, vq_model, fid_calculator, rank, device, ptdtype, logger)
                if rank == 0:
                    log_infos(eval_outputs, train_steps, "test", writer, epoch=epoch)
        
        if is_final:
            break


    vq_model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    dist.destroy_process_group()
    wandb.finish()

    if rank == 0 and writer is not None:
        writer.close()


def save_ckpt(ckpt_prefix, args, vq_model, optimizer, vq_loss, optimizer_disc, train_steps, ema, checkpoint_dir, logger, cloud_checkpoint_dir, rank):
    # Save checkpoint:
    if rank == 0:
        if args.compile:
            model_weight = vq_model.module._orig_mod.state_dict()
        else:
            model_weight = vq_model.module.state_dict()  
        checkpoint = {
            "model": model_weight,
            "optimizer": optimizer.state_dict(),
            "discriminator": vq_loss.module.discriminator.state_dict(),
            "optimizer_disc": optimizer_disc.state_dict(),
            "steps": train_steps,
            "args": args
        }
        if args.ema:
            checkpoint["ema"] = ema.state_dict()
        
        # always save locally
        checkpoint_path = f"{checkpoint_dir}/{ckpt_prefix}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if args.cloud_save_path is not None:
            cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, cloud_checkpoint_path)
            logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
    dist.barrier()

def evaluate_one_round(test_loader, vq_model, fid_calculator, rank, device, ptdtype, logger, img_num=8):
    vq_model.eval()
    vq_model.module.quantize.reset_codebook_usage()
    img_fake_stats = None
    img_running_real_stats = None

    pbar = tqdm(test_loader, desc='eval', leave=True) if rank==0 else test_loader
    for i, (x, y) in enumerate(pbar):
        imgs = x.to(device, non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
            (vqkd_recon, reconstructions), codebook_loss = vq_model(imgs)
            reconstructions = ((reconstructions + 1.0) / 2.0).clamp(0.0, 1.0)
            imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
            
            img_fake_stats = fid_calculator.get_feature_stats_for_batch(reconstructions, img_fake_stats)
            img_running_real_stats = fid_calculator.get_feature_stats_for_batch(imgs, img_running_real_stats)

            # return images for wandb and tensorboard logging
            if i == len(pbar) - 1:
                images_gt = vutils.make_grid(imgs[:img_num], nrow=4, padding=2)  # shape [C,H,W]
                images_recon = vutils.make_grid(reconstructions[:img_num], nrow=4, padding=2)

    if img_fake_stats is not None:
        try:
            real = img_running_real_stats
            fid = fid_calculator.calculate_fid_smart(
                img_fake_stats, real, bs=32, cache_stats=(rank == 0)
            )
        except Exception as e:
            logger(f'FID calculation failed: {e}')

    infos = {
        'fid': fid,
        'images_gt': images_gt,
        'images_recon': images_recon
    }
    codebook_usage = codebook_loss[3]
    if isinstance(codebook_usage, (Tuple, List)):
        for i, usage in enumerate(codebook_usage):
            infos[f"usage{i}"] = usage
        logger.info(f"(Evaluation) FID: {fid:.4f}, Usage: {[f'{i:.4f}' for i in codebook_usage]}")
    else:
        infos.update(codebook_usage=codebook_usage)
        logger.info(f"(Evaluation) FID: {fid:.4f}, Codebook_Usage: {codebook_loss[3]:.4f}")

    vq_model.module.quantize.reset_codebook_usage()
    vq_model.train()

    return infos



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/path/to/your/dataset')
    parser.add_argument("--val-data-path", type=str, default='/path/to/your/val_dataset')
    
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default=None, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="EvoTok")
    parser.add_argument("--teacher", type=str, choices=["clipb_224", "vitamin_xlarge_256", "siglip_384", "siglip_256"], default="clipb_224", help="the semantic teacher, important")

    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--vq-warmup", type=int, default=0, help="whether using warmup for ema training")

    parser.add_argument("--restart-unused-codes", action='store_true', help="whether restarting unused codes.")
    parser.add_argument("--vqkd-loss", type=str, default="cosine", help="vqkd loss, e.g., cosine, cosine+mse")
    parser.add_argument("--vqkd-depth", type=int, default=8, help="residual depth for semantic feature")
    parser.add_argument("--vqgan-depth", type=int, default=1, help="residual depth for pixel feature")
    parser.add_argument("--codebook-size", type=int, default=8192, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for pixel vector quantization")
    parser.add_argument("--vqkd-weight", type=float, default=1.0, help="vqkd loss weight for vector quantization")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[224, 256, 384], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=512) 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--ckpt-last-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--by-epoch", action='store_true', help="wether to save ckpts and eval by epoch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)