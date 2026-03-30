import logging
import torch.distributed as dist
import os
import json
import wandb
import requests
from filelock import FileLock
from torch.utils.tensorboard import SummaryWriter
import torch


def log_infos(log_dict, step, prefix="", writer=None, epoch=None):
    image_dict = {k: v for k, v in log_dict.items() if k.startswith("images")}
    scalar_dict = {k: v for k, v in log_dict.items() if not k.startswith("images")}

    # TensorBoard logging
    if writer is not None:
        for key, val in scalar_dict.items():
            tag = f"{prefix}/{key}"
            writer.add_scalar(tag, val, global_step=step)
            if epoch is not None:
                writer.add_scalar(f"epoch-{tag}", val, global_step=epoch)

        for key, val in image_dict.items():
            if isinstance(val, torch.Tensor) and val.ndim == 3 and val.shape[0] == 3:
                tag = f"{prefix}/{key}"
                writer.add_image(tag, val, global_step=step)
                if epoch is not None:
                    writer.add_image(f"epoch-{tag}", val, global_step=epoch)

    # WandB logging
    wandb_log_dict = {}

    for key, val in scalar_dict.items():
        wandb_log_dict[f"{prefix}/{key}"] = val
    for key, val in image_dict.items():
        print(val.shape)
        if isinstance(val, torch.Tensor) and val.ndim == 3 and val.shape[0] == 3:
            val = val.to(torch.float32)
            wandb_log_dict[f"{prefix}/{key}"] = wandb.Image(val)

    if epoch is not None:
        wandb_log_dict[f"{prefix}/epoch"] = epoch

    wandb.log(wandb_log_dict, step=step)



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def enable_tensorboard(save_dir):
    if save_dir is not None:
        writer = SummaryWriter(os.path.join(save_dir, f'tensorboard'))
    else:
        writer = None
    return writer

def check_website_access_bool(url):
    try:
        response = requests.get(url, timeout=3)
        # Check if the response code is 200 or 403
        if response.status_code in [200, 403]:
            return True
        else:
            return False
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        # Any exception means the site is not accessible within the parameters
        return False
    
def enable_wandb(project, exp_name, config, save_dir, wandb_run_id=None, offline_mode=False):
    os.environ['WANDB_DIR'] = save_dir
    os.environ['WANDB_NAME'] = exp_name

    if wandb_run_id is not None:
        init_kwargs = {'id': wandb_run_id, 'resume': 'allow'}
    else:
        init_kwargs = {'resume': 'allow'}

    if offline_mode or not check_website_access_bool('https://wandb.ai'):
        print('Wandb website not accessible. Running in offline mode.')
        init_kwargs['mode'] = 'offline'
        this_run_metadata = {
            'project': project,
            'dir': os.path.abspath(os.path.join(save_dir, 'wandb'))
        }
        run_metadata_json_path = './wandb_run_metadata.json'
        with FileLock(run_metadata_json_path + '.lock'):
            if os.path.exists(run_metadata_json_path):
                with open(run_metadata_json_path, 'r') as f:
                    run_metadata_all = json.load(f)
            else:
                run_metadata_all = []

            run_metadata_all.append(this_run_metadata)
            with open(run_metadata_json_path, 'w') as f:
                json.dump(run_metadata_all, f, indent=4)

    wandb.init(
        project=project, 
        config=config,
        **init_kwargs
    )
    print(f"Successfully init wandb!")