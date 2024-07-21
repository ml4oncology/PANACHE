import torch.utils.data as data
import argparse
import torch.nn as nn
import os
import torch
from tqdm import tqdm
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import collections
from sksurv.metrics import concordance_index_censored as concordance_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class microscope_data(data.Dataset):
    """
    A Pytorch dataset that returns a tensor of concatenated HIPT 4k features

    Inputs:

        libraryfile: The dictionary with the following keys:
            slide_id: slide ids, len is number of WSI
            grid: a list containing sub-list of tile coords (x,y) each sublist belongs to a WSI, len is number of WSI
        ftr_path: path to folder containing slide_id subfolders, each subfolder contains saved tile features

    """

    def __init__(self, libraryfile="", ftr_path=""):
        lib = torch.load(libraryfile)

        print("Slides: {}".format(len(lib["slide_id"])))

        self.slide_id = lib["slide_id"]
        self.coords = lib["grid"]
        self.ftr_path = ftr_path

        # concatenate all 4k ftrs for each WSI
        self.ftrs_4k_cat = []
        for i, slide_id in enumerate(self.slide_id):
            ftrs_4k = []
            for coord in self.coords[i]:
                ftr_name = slide_id + "_" + str(coord[0]) + "_" + str(coord[1]) + ".pt"
                ftrs_4k.append(
                    torch.load(
                        os.path.join(self.ftr_path, slide_id, ftr_name),
                        map_location=device,
                    )
                )
            self.ftrs_4k_cat.append(torch.cat(ftrs_4k))

    def __getitem__(self, index):
        slide_id = self.slide_id[index]
        ftrs_4k_cat = self.ftrs_4k_cat[index]
        # mean pooling of ftrs, [1 x #4096tiles x 192] -> [1 x 192]
        ftrs_mean = torch.mean(ftrs_4k_cat, 0)

        return ftrs_mean

    def __len__(self):
        return len(self.slide_id)


class Model3Layers(nn.Module):
    def __init__(self, dropout_prob):
        super(Model3Layers, self).__init__()
        self.hidden1 = nn.Linear(192, 96).to(device)
        self.hidden2 = nn.Linear(96, 48).to(device)
        self.regression = nn.Linear(48, 1).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)

    def forward(self, ftrs):
        x = self.hidden1(ftrs)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        return self.regression(x)


def get_dataloader(dataset, batch_size, pin_memory=False, num_workers=0):
    g = torch.Generator()
    g.manual_seed(416)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        generator=g,
    )
    print(f"\nNumber of samples: {len(dataloader.dataset)}")
    print(f"Number of batches on device: {len(dataloader)}")
    return dataloader


def get_predictions(model, dset, args):

    loader = get_dataloader(dset, batch_size=1, num_workers=args.workers)
    print("\nRetreving model predictions")
    preds = []

    model.eval()
    with torch.no_grad():
        batches = tqdm(loader, total=len(loader), desc="Tuning loss")
        for i, ftrs in enumerate(batches):
            pred = model(ftrs)
            preds.append(pred.detach()[:, 0].clone())

    # write to output csv
    preds = torch.cat(preds, 0)
    writer = open(args.output_csv, "a+")

    for i, pred in enumerate(preds):
        writer.write(f"{dset.slide_id[i]},{pred.item()}\n")
    writer.close()


def main(args):

    print("\nLoading 4k ftrs dataset")
    dset = microscope_data(args.wsi_dict, args.ftr_directory)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Load from checkpoint
    if args.checkpoint_file:
        model = Model3Layers(args.dropout_prob)
        model = nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(args.checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # csv headers
    writer = open(args.output_csv, "a+")
    writer.write("slide_id, pred\n")
    writer.close()
    get_predictions(model, dset, args)
    print(f"All predictions saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT for histology")
    parser.add_argument(
        "--wsi_dict",
        type=str,
        help="dictionary file containing slide_ids and coordinates of tiles",
    )
    parser.add_argument(
        "--ftr_directory",
        type=str,
        default="",
        help="path to the WSI 4k tile ftrs in subdirectories",
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="number of data loading workers"
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default="", help="model state_dict location"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="full path to output csv for logging WSI predictions",
    )
    parser.add_argument("--seed", default=416, type=int, help="random seed")
    parser.add_argument(
        "--dropout_prob",
        default=0.1,
        type=float,
        help="probability of dropout in hidden layer",
    )
    args = parser.parse_args()
    main(args)
