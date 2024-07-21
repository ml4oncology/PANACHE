import argparse
import os
import openslide as slide
import numpy as np
from skimage.filters import threshold_otsu
import torch
from tqdm import tqdm
from collections import defaultdict

"""
* assuming same magnification
Inputs:
    - input_slide_dir: path to directory containing svs files - svs files should have format <slide_id>.svs
    - magnification: magnification level of WSIs
Outputs:
    - output_jpg_dir: path to directory to save jpg tiles
    - output_dict_dir: path to directory to save dictionary with slide ids and tile coordinates
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_slide_dir", type=str, help="path to directory containing .svs files"
)
parser.add_argument(
    "--magnification", default=20, type=int, help="magnification of slides"
)
parser.add_argument(
    "--tile_size", default=4096, type=int, help="dimension of tiles in pixels"
)
parser.add_argument(
    "--output_jpg_dir", type=str, help="path to directory to save jpg tiles"
)
parser.add_argument(
    "--output_dict_dir",
    type=str,
    help="path to directory to save dictionary with slide ids and tile coordinates",
)
parser.add_argument(
    "--tile_tissue_proportion",
    default=0.2,
    type=float,
    help="minimum % of tissue required in tile to include in dataset",
)
args = parser.parse_args()

if args.magnification not in [20, 40]:
    parser.error("Magnification must be either 20 or 40")
if not os.path.exists(args.output_jpg_dir):
    os.makedirs(args.output_jpg_dir)
if not os.path.exists(args.output_dict_dir):
    os.makedirs(args.output_dict_dir)


def get_coords(file_path, mag, window_size, otsu_prop):
    wsi = slide.open_slide(file_path)
    if (
        mag == 40
    ):  # tiles are twice as large but eventually scaled down to args.tile_size x args.tile_size
        window_size = window_size * 2
    max_x, max_y = wsi.level_dimensions[0]

    # get scaled-down WSI for binary mask
    # scale factor 256
    gray_thumb = wsi.get_thumbnail((max_x // 256, max_y // 256)).convert("L")
    tiles_x, tiles_y = max_x // window_size, max_y // window_size
    print("Horizontal, pixels: {}, tiles: {}".format(max_x, tiles_x))
    print("Vertical, pixels: {}, tiles: {}".format(max_y, tiles_y))
    wsi_array = np.asarray(gray_thumb)
    threshold = threshold_otsu(wsi_array)
    binary = wsi_array < threshold
    n_256 = window_size // 256  # number of 256 dim patches within a tile
    coords = []
    for i in range(tiles_y):
        for j in range(tiles_x):
            if (
                binary[i * n_256 : (i + 1) * n_256, j * n_256 : (j + 1) * n_256].sum()
                / (n_256**2)
                >= otsu_prop
            ):
                coords.append((j * window_size, i * window_size))
    return wsi, coords


def save_jpgs(openslide_obj, slide_id, coords, tile_size, output_dir, mag, level=0):
    """
    Creates subdirectories by slide_id in output_dir, saves tile jpgs in subdirectories for each WSI
    """
    if mag == 40:
        coord_size = tile_size * 2
    else:
        coord_size = tile_size
    for i in range(len(coords)):
        jpg_path = os.path.join(
            output_dir, slide_id, f"{slide_id}_{coords[i][0]}_{coords[i][1]}.jpg"
        )
        if not os.path.exists(os.path.join(output_dir, slide_id)):
            os.mkdir(os.path.join(output_dir, slide_id))
        if os.path.exists(jpg_path):
            continue
        img = openslide_obj.read_region(
            coords[i], level, (coord_size, coord_size)
        ).convert("RGB")
        if mag == 40:
            img = img.resize((tile_size, tile_size))
        img.save(jpg_path)


def main():
    all_wsis = os.listdir(args.input_slide_dir)
    wsi_dict = defaultdict(list)
    for wsi in tqdm(all_wsis, total=len(all_wsis)):
        if os.path.isfile(os.path.join(args.input_slide_dir, wsi)) and "svs" in wsi:
            slide_id = wsi.split(".svs")[0]
            wsi, coords = get_coords(
                os.path.join(args.input_slide_dir, wsi),
                args.magnification,
                args.tile_size,
                args.tile_tissue_proportion,
            )
            # save slide id and coords in dictionary, save jpgs
            if len(coords) >= 1:
                save_jpgs(
                    wsi,
                    slide_id,
                    coords,
                    args.tile_size,
                    args.output_jpg_dir,
                    args.magnification,
                )
                wsi_dict["slide_id"].append(slide_id)
                wsi_dict["grid"].append(coords)
            else:
                print(f"Excluding {slide_id}, not enough tiles")
    torch.save(wsi_dict, os.path.join(args.output_dict_dir, "wsi_coord_dict.pth"))


if __name__ == "__main__":
    main()
