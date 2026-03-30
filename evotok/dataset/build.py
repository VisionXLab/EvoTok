from dataset.imagenet import build_imagenet
from dataset.coco import build_coco
from dataset.concat_folder_dataset import build_multiple_dataset

def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'multiple':
        return build_multiple_dataset(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')