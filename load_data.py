import os
import time
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate
from transforms import get_mixup_cutmix
import presets

def count_folders(directory_path):
    folder_count = 0
    for _, dirnames, _ in os.walk(directory_path):
        folder_count += len(dirnames)
    return folder_count

def load_data(args):
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")
    num_classes = count_folders(traindir)
    print("## Train Dir:", traindir)
    print("## Val Dir:", valdir)    
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

    # We need a default value for the variables below because args may come
    # from train_quantization.py which doesn't define them.
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=args.backend,
            use_v2=args.use_v2,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
        use_v2=args.use_v2,
    )

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        collate_fn=default_collate,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=test_sampler,
        pin_memory=True,
    )

    return data_loader, data_loader_test, train_sampler, test_sampler, num_classes
