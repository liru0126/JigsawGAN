from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.jigsaw_data_loader import JigsawDataset


def get_dataset_info(txt_list):
    images_list = open(txt_list).read().splitlines()

    file_names = []
    for row in images_list:
        file_names.append(row)

    return file_names


def get_train_dataloader(args):
    # datasets = []
    # val_datasets = []
    img_transformer, img_tensor, tile_transformer = get_train_transformers(args)

    name_train = get_dataset_info(args.data_path + 'train.list')
    name_val = get_dataset_info(args.data_path + 'test.list')

    train_dataset = JigsawDataset(name_train, data_path=args.data_path, patches=True, img_transformer=img_transformer, img_tensor=img_tensor, tile_transformer=tile_transformer, grid_size=args.grid_size, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    val_dataset = JigsawDataset(name_val, data_path=args.data_path, patches=True, img_transformer=img_transformer, img_tensor=img_tensor,tile_transformer=tile_transformer, grid_size=args.grid_size, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader, len(train_dataset)


def get_train_transformers(args):
    img_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
    ])
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tile_tr = []
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    return img_transform, img_tensor, transforms.Compose(tile_tr)



