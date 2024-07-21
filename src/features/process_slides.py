import argparse
import os
import subprocess

"""
This file selects all slides by cohort and runs the extract_features.py script on them
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_svs_directory", type=str, help="input directory of .svs whole slide images"
)
parser.add_argument("--clinical_file", type=str, help="clinical file path")
parser.add_argument("--cohort", type=str, help="cohort, one of tcga, cia, icgc")
parser.add_argument(
    "--output_dict_directory",
    type=str,
    help="output directory for dictionary and summary csv",
)
parser.add_argument(
    "--output_jpg_directory", type=str, help="output directory for jpg tiles"
)
parser.add_argument(
    "--output_tensor_directory",
    default="n/a",
    type=str,
    help="output directory for tile tensors",
)
parser.add_argument(
    "--magnification", default=20, type=str, help="magnification, default=20"
)
parser.add_argument(
    "--number_jobs",
    default=1000,
    type=str,
    help="number of jobs queueing simultaneously on UGE",
)
parser.add_argument("--log_dir", type=str, help="location for error and output files")
parser.add_argument("--size", default=224, type=int, help="size of tile in pixels")
parser.add_argument(
    "--tile_tissue_proportion",
    default=0,
    type=float,
    help="minimum percentage of the area that should be tissue in a 3x3 tile grid centered at current tile",
)
args = parser.parse_args()

input_svs_directory = args.input_svs_directory
output_dict_directory = args.output_dict_directory
output_jpg_directory = args.output_jpg_directory
output_tensor_directory = args.output_tensor_directory
clinical_file = args.clinical_file
cohort = args.cohort
magnification = args.magnification
log_dir = args.log_dir
size = args.size
tile_tissue_proportion = args.tile_tissue_proportion

extract_features_script = "./src/features/extract_features.py"

if cohort not in ["tcga", "cia", "icgc"]:
    parser.error("Cohort must be one of tcga, cia, icgc")


def get_slides(path):
    """
    Input:
    path: directory with slides as .svs files in subdirectories

    Output:
    list of the paths to all slides in the directory
    """
    svs = []
    for subdir_path, _, filenames in os.walk(path):
        filenames.sort()
        for name in filenames:
            if name.endswith(".svs"):
                svs_path = os.path.join(subdir_path, name)
                svs.append(svs_path)
    return svs


def get_slide_id(file_path, cohort):
    """
    Gets the slide id from file_path based on cohort
    For cia, first 12 characters in filename
    For tcga, first 16
    """
    if cohort == "cia":
        slide_id = file_path.split("/")[-1][:12]
    elif cohort == "tcga" or cohort == "tcga_breast":
        slide_id = file_path.split("/")[-1][:16]
    elif cohort == "icgc":
        slide_id = file_path.split("/")[-1][:-4]
    else:
        slide_id = "error"
    return slide_id


svs = get_slides(input_svs_directory)

for i in range(len(svs)):
    print("{}/{} slides, now running {}".format(i + 1, len(svs), svs[i]), flush=True)
    subprocess.call(
        [
            f"python {extract_features_script} --input_slide {svs[i]} --clinical_file {clinical_file} --output_dict_directory {output_dict_directory} --output_jpg_directory {output_jpg_directory} --cohort {cohort} --magnification {magnification} --size {size} --save_thumb --save_tile_jpgs --tile_tissue_proportion {tile_tissue_proportion}",
        ]
    )
