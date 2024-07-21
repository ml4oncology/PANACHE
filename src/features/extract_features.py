import argparse
import os
import openslide as slide
import numpy as np
from skimage.filters import threshold_otsu
import torch
import pandas as pd
import math
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_slide", type=str, help="path to input .svs whole slide image"
)
parser.add_argument("--clinical_file", type=str, help="file with clinical info")
parser.add_argument(
    "--output_dict_directory",
    type=str,
    help="output directory for dictionary and summary csv",
)
parser.add_argument(
    "--output_jpg_directory", type=str, help="output directory for tile jpgs"
)
parser.add_argument("--cohort", type=str, help="cohort, one of tcga, cia, icgc")
parser.add_argument(
    "--magnification",
    default=20,
    type=int,
    help="magnification of slides, script rescales to 20x default",
)
parser.add_argument(
    "--save_tile_jpgs", action="store_true", help="save each tile as a jpeg"
)
parser.add_argument(
    "--save_tile_tensors",
    action="store_true",
    help="save tiles as transformed tensors to speed up dataloading",
)
parser.add_argument("--size", default=224, type=int, help="size of tile in pixels")
parser.add_argument(
    "--tile_tissue_proportion",
    default=0.2,
    type=float,
    help="minimum percentage of the area that should be tissue in a 3x3 tile grid centered at current tile",
)
parser.add_argument(
    "--save_thumb", action="store_true", help="save slide as a thumb jpeg"
)
parser.add_argument(
    "--overwrite", action="store_true", help="whether or not to overwrite existing jpgs"
)

args = parser.parse_args()
if args.cohort not in ["tcga", "cia", "icgc"]:
    parser.error("Cohort must be one of tcga, cia, icgc")

if args.magnification not in [20, 40]:
    parser.error("Magnification must be either 20 or 40")

input_slide = args.input_slide
print(input_slide)
clinical = args.clinical_file
output_dict_directory = args.output_dict_directory
output_jpg_directory = args.output_jpg_directory
cohort = args.cohort
magnification = args.magnification
save_tile_jpgs = args.save_tile_jpgs
save_tile_tensors = args.save_tile_tensors
size = args.size
tile_tissue_proportion = args.tile_tissue_proportion
save_thumb = args.save_thumb

if not os.path.exists(output_dict_directory):
    os.makedirs(output_dict_directory)
if output_jpg_directory:
    if not os.path.exists(output_jpg_directory):
        os.makedirs(output_jpg_directory)


def get_tcga_ids(file_path):
    """
    Gets the slide and patient id, which is the first 16 and 12 characters of the filename
    """
    return file_path.split("/")[-1][:16], file_path.split("/")[-1][:12]


def get_cia_ids(file_path):
    """
    Gets the slide id and patient id, which is the first 12 and 9 characters of the filename
    """
    return file_path.split("/")[-1][:12], file_path.split("/")[-1][:9]


def get_icgc_ids(file_path):
    """
    Gets slide id and patient id, which is the last 4 and first 4 characters of the filename
    """
    return file_path.split("/")[-1][:-4], file_path.split("/")[-1][:4]


def get_coords(file_path, window_size=size, mag=20, otsu_prop=0.6):
    """
    Gets PIL-style image coordinates (Cartesian) of the top-left position of tiles in the foreground
        in order to remove the background whitespace in the tile

    Inputs
        file_path: Path to the input slide
        window_size: Pixels in the tile, i.e. 224 for Resnet
        mag: magnification of the slide
        otsu_prop: proportion of the tile in the foreground to be retained

    Outputs
        wsi: openslide object
        max_x: x axis pixels in PIL
        max_y: y axis pixels in PIL
        coords: a list of the coordinates in the foreground, given as PIL coordinates (x, y)

        Note:
        Given a set of cartesian coordinates (x, y) (positive y-axis is flipped),
                Numpy: indexes row first and then column ie. 2d_array[y][x]
                PIL: normal cartesian coordinates ie. (x, y) -> used for openslide objects
    """

    wsi = slide.open_slide(file_path)
    if mag == 40:  # Tiles are twice as large but scaled down to args.sizexargs.size
        window_size = window_size * 2
    max_x, max_y = wsi.level_dimensions[0]
    if window_size == 4096 or window_size == 8192:
        # Each pixel corresponds to one 256x256 tile (16x16 pixels now correspond to one 4096x4096 tile)
        gray_thumb = wsi.get_thumbnail((max_x // 256, max_y // 256)).convert("L")
        tiles_x, tiles_y = max_x // window_size, max_y // window_size
    else:
        # Each pixel corresponds to one 224x224 tile or one 448x448 tile if mag is 40
        gray_thumb = wsi.get_thumbnail(
            (max_x // window_size, max_y // window_size)
        ).convert("L")
        tiles_x, tiles_y = gray_thumb.size
    print("Horizontal, pixels: {}, tiles: {}".format(max_x, tiles_x))
    print("Vertical, pixels: {}, tiles: {}".format(max_y, tiles_y))

    wsi_array = np.asarray(gray_thumb)
    threshold = threshold_otsu(wsi_array)
    binary = wsi_array < threshold  # Exclude if lighter than threshold
    coords = []

    if window_size < 4096:
        # exclude outer rim of tiles
        binary[[0, -1], :] = False
        binary[:, [0, -1]] = False
        # examine 3x3 tile area, only include center tile if >=otsu_prop tiles contain tissue
        for i in range(1, tiles_y - 1):
            for j in range(1, tiles_x - 1):
                if (
                    binary[i, j]
                    and binary[i - 1 : i + 2, j - 1 : j + 2].sum() / 9 >= otsu_prop
                ):  # originally 0.7 for small tiles
                    coords.append((j * window_size, i * window_size))
                elif binary[i, j]:  # if tile contains tissue
                    coords.append((j * window_size, i * window_size))

    # Keep if tile contains majority foreground
    if window_size == 4096 or window_size == 8192:
        n_256 = (
            window_size // 256
        )  # number of 256 dim patches within a tile of dimension window_size
        for i in range(tiles_y):
            for j in range(tiles_x):
                if (
                    binary[
                        i * n_256 : (i + 1) * n_256, j * n_256 : (j + 1) * n_256
                    ].sum()
                    / (n_256**2)
                    >= otsu_prop
                ):
                    coords.append((j * window_size, i * window_size))

    print("{}/{} tiles remaining after filter".format(len(coords), tiles_x * tiles_y))

    return wsi, max_x, max_y, coords


def save_tiles(openslide_obj, coordinates, slide_name, size=size, mag=20, level=0):
    """
    Saves a tile as a jpg

    Inputs:
        openslide_obj: A wsi openslide object saved with open_slide
        coordinates: A list of tuple coordinates meeting the criteria
        slide_id: Slide id from get_slide_id
        out_path: Prefix for folder to save images
        level: Level on openslide image, default 0
        size: Number of pixels
        mag: The magnification. If 40x, then you must load double the size and rescale down.
    """
    if mag == 40:
        coord_size = size * 2
    else:
        coord_size = size
    for i in range(len(coordinates)):
        if save_tile_jpgs:
            print("Saving tiles: {}, {} as jpg".format(i, coordinates[i]))
            jpg_path = os.path.join(
                output_jpg_directory,
                slide_name,
                "{}_{}_{}.jpg".format(slide_name, coordinates[i][0], coordinates[i][1]),
            )
            if not os.path.exists(os.path.join(output_jpg_directory, slide_name)):
                os.mkdir(os.path.join(output_jpg_directory, slide_name))
            if (
                not os.path.exists(jpg_path) or args.overwrite
            ):  # create or overwrite jpg
                img = openslide_obj.read_region(
                    coordinates[i], level, (coord_size, coord_size)
                ).convert("RGB")
                if (
                    mag == 40
                ):  # resize from 448x448 to 224x224, or resize from 8192x8192 to 4096x4096
                    img = img.resize((size, size))
                img.save(jpg_path)

        if save_tile_tensors:
            print("Saving tiles: {}, {} as tensor".format(i, coordinates[i]))
            tensor_path = os.path.join(
                args.output_tensor_directory,
                "{}_{}_{}".format(slide_name, coordinates[i][0], coordinates[i][1]),
            )
            if not os.path.exists(args.output_tensor_directory):
                os.mkdir(args.output_tensor_directory)
            if not os.path.exists(tensor_path):  # don't overwrite existing tensor
                device = torch.device("cuda:0" if torch.cuda.is_available() else None)

                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                color_jitter = transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                )
                hflip = transforms.RandomHorizontalFlip(p=0.5)
                vflip = transforms.RandomVerticalFlip(p=0.5)
                cpu_transforms = transforms.Compose(
                    [color_jitter, hflip, vflip, transforms.ToTensor()]
                )

                img = cpu_transforms(img)
                if device:
                    img = img.to(device)  # perform normalization on gpu if avail
                img = normalize(img)
                torch.save(img, tensor_path)


def save_thumb(openslide_obj, slide_name, out_path, size=224, mag=20, level=0):
    """
    Saves a thumbnail of the slide as a jpg in the jpg directory
    """
    max_x, max_y = openslide_obj.level_dimensions[0]
    thumb = openslide_obj.get_thumbnail((max_x // size, max_y // size))
    print(out_path, slide_name)
    thumb_path = os.path.join(out_path, slide_name, "{}_thumb.jpg".format(slide_name))
    if not os.path.exists(os.path.join(out_path, slide_name)):
        os.mkdir(os.path.join(out_path, slide_name))
    if not os.path.exists(thumb_path):
        thumb.save(thumb_path)


def exclude_slide(slide_dict):
    """
    Excludes slide if it is not a good candidate for the dataset

    Inputs:
        slide_dict: the dictionary containing features extracted from the clinical file and the tile coordinates
    Outputs:
        True or False: whether the slide should be excluded from the dataset or not
    """
    if slide_dict["time"] is None or math.isnan(slide_dict["time"]):
        print(
            "Excluding {} because survival is {}".format(
                slide_dict["slide_id"], slide_dict["time"]
            ),
            flush=True,
        )
        return True
    if slide_dict["vital"] is None or math.isnan(slide_dict["vital"]):
        print(
            "Excluding {} because event is {}".format(
                slide_dict["slide_id"], slide_dict["vital"]
            ),
            flush=True,
        )
        return True
    if size < 4096 and len(slide_dict["grid"]) < 100:
        print(
            "Excluding {} because only {} tiles".format(
                slide_dict["slide_id"], len(slide_dict["grid"])
            ),
            flush=True,
        )
        return True
    if size >= 4096 and len(slide_dict["grid"]) < 1:
        print(
            "Excluding {} because only {} tiles".format(
                slide_dict["slide_id"], len(slide_dict["grid"])
            ),
            flush=True,
        )
        return True
    return False


def main():
    # extract clinical information
    if cohort == "tcga":
        slide_id, patient_id = get_tcga_ids(input_slide)
        header = pd.read_csv(clinical, sep="\t", nrows=1)
        df = pd.read_csv(
            clinical,
            skiprows=3,
            sep="\t",
            na_values=["[Not Available]", "[Unknown]", "[Not Applicable]"],
            names=header.columns,
        )
        grade = np.select(
            (
                (df[df["bcr_patient_barcode"] == patient_id]["tumor_grade"] == "G1"),
                (df[df["bcr_patient_barcode"] == patient_id]["tumor_grade"] == "G2"),
                (df[df["bcr_patient_barcode"] == patient_id]["tumor_grade"] == "G3"),
                (df[df["bcr_patient_barcode"] == patient_id]["tumor_grade"] == "G4"),
            ),
            (1, 2, 3, 4),
            default=None,
        ).flat[0]
        time = (
            df[df["bcr_patient_barcode"] == patient_id][
                ["last_contact_days_to", "death_days_to"]
            ]
            .min(axis=1, skipna=True)
            .iat[0]
        )
        vital = np.select(
            (
                (df[df["bcr_patient_barcode"] == patient_id]["vital_status"] == "Dead"),
                (
                    df[df["bcr_patient_barcode"] == patient_id]["vital_status"]
                    == "Alive"
                ),
            ),
            (1, 0),
            default=None,
        ).flat[0]
        t_stage = df[df["bcr_patient_barcode"] == patient_id][
            "ajcc_tumor_pathologic_pt"
        ].iat[0]
        n_stage = df[df["bcr_patient_barcode"] == patient_id][
            "ajcc_nodes_pathologic_pn"
        ].iat[0]
        r_status = df[df["bcr_patient_barcode"] == patient_id]["residual_tumor"].iat[0]
        age = df[df["bcr_patient_barcode"] == patient_id][
            "age_at_initial_pathologic_diagnosis"
        ].iat[0]
        m_stage = df[df["bcr_patient_barcode"] == patient_id][
            "ajcc_metastasis_pathologic_pm"
        ].iat[0]
        if not np.isnan(age):
            age = int(age)
        sex = df[df["bcr_patient_barcode"] == patient_id]["gender"].iat[0]
        if isinstance(sex, str):
            sex = sex.lower()

    elif cohort == "cia":
        slide_id, patient_id = get_cia_ids(input_slide)
        df = pd.read_csv(clinical)
        grade = df[df["slide_id"] == slide_id]["grade"].iat[0]
        time = df[df["slide_id"] == slide_id]["time"].iat[0]
        vital = df[df["slide_id"] == slide_id]["vital"].iat[0]
        t_stage = df[df["slide_id"] == slide_id]["tumor_stage_pathological"].iat[0]
        n_stage = df[df["slide_id"] == slide_id][
            "pathologic_staging_regional_lymph_nodes_pN"
        ].iat[0]
        r_status = df[df["slide_id"] == slide_id]["residual_tumor"].iat[0].split(":")[0]
        m_stage = df[df["slide_id"] == slide_id][
            "clinical_staging_distant_metastasis_cM"
        ].iat[0]
        age = df[df["slide_id"] == slide_id]["age"].iat[0]
        if not np.isnan(age):
            age = int(age)
        sex = df[df["slide_id"] == slide_id]["gender"].iat[0]
        if isinstance(sex, str):
            sex = sex.lower()
        print("done")

    elif cohort == "icgc":
        slide_id, patient_id = get_icgc_ids(input_slide)
        file_name = slide_id + ".svs"
        df = pd.read_csv(clinical)
        grade = df[df["file"] == file_name]["grade"].iat[0]
        time = df[df["file"] == file_name]["os"].iat[0]
        vital = df[df["file"] == file_name]["vital"].iat[0]
        #!
        if isinstance(df[df["file"] == file_name]["stage"].iat[0], float):
            t_stage = np.nan
            n_stage = np.nan
            m_stage = np.nan
        else:
            t_stage = df[df["file"] == file_name]["stage"].iat[0][:2]
            n_stage = df[df["file"] == file_name]["stage"].iat[0][2:4]
            m_stage = df[df["file"] == file_name]["stage"].iat[0][-2:]

        r_status = np.nan
        age = df[df["file"] == file_name]["age_enrol"].iat[0]
        if not np.isnan(age):
            age = int(age)
        sex = df[df["file"] == file_name]["sex"].iat[0]
        if isinstance(sex, str):
            sex = sex.lower()

    # apply filter, tile slides, and save tiles
    wsi, max_x, max_y, coords = get_coords(
        input_slide,
        window_size=size,
        mag=magnification,
        otsu_prop=tile_tissue_proportion,
    )
    # Output dictionary to use to load images and labels
    pdac_dict = {}
    pdac_dict["slide_path"] = input_slide
    pdac_dict["magnification"] = magnification
    pdac_dict["cohort"] = cohort
    pdac_dict["slide_id"] = slide_id
    pdac_dict["patient_id"] = patient_id
    pdac_dict["grid"] = coords
    pdac_dict["grade"] = grade
    pdac_dict["time"] = time
    pdac_dict["vital"] = vital
    pdac_dict["t_stage"] = t_stage
    pdac_dict["n_stage"] = n_stage
    pdac_dict["r_status"] = r_status
    pdac_dict["age"] = age
    pdac_dict["sex"] = sex
    pdac_dict["m_stage"] = m_stage

    print(pdac_dict)

    if not exclude_slide(pdac_dict):
        if save_thumb:
            save_thumb(wsi, slide_id, output_jpg_directory, 224, mag=magnification)
        if save_tile_jpgs or save_tile_tensors:
            save_tiles(wsi, coords, slide_id, size, mag=magnification)
        wsi.close()  #!
        output_csv = open(
            os.path.join(output_dict_directory, slide_id + "_slides.csv"), "w"
        )
        output_csv.write("{},{},{},{}\n".format(input_slide, patient_id, max_x, max_y))
        output_csv.close()
        print(f"Number of tiles for {slide_id}: {len(coords)}")
        torch.save(
            pdac_dict, os.path.join(output_dict_directory, slide_id + "_pdac_dict.pth")
        )


if __name__ == "__main__":
    main()
