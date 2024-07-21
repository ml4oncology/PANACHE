import argparse
import os
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dict_directory",
    type=str,
    help="Directory with all the dictionaries to be merged",
)
parser.add_argument(
    "--output_train_dict", type=str, help="Training dictionary output full path"
)
parser.add_argument(
    "--output_tune_dict", type=str, help="Tuning dictionary output full path"
)
parser.add_argument(
    "--output_test_dict", type=str, help="Test dictionary output full path"
)
parser.add_argument(
    "--cohort_split",
    action="store_true",
    help="Split development and test sets by cohort rather than random, false if unspecified",
)
parser.add_argument(
    "--train_cohorts",
    type=str,
    default="tcga,cia,icgc",
    help="names separated by commas, if not cohort split, all dictionaries contain these cohorts only",
)
parser.add_argument(
    "--test_cohorts", type=str, default="oicr", help="names separated by commas"
)
parser.add_argument(
    "--tune_proportion",
    type=float,
    help="Proportion of examples for tuning",
    default=0.25,
)
parser.add_argument(
    "--test_proportion",
    type=float,
    help="Proportion of examples for testing",
    default=0.25,
)
parser.add_argument(
    "--exclude_list", type=str, help="csv file with patient_id column to exclude on"
)
args = parser.parse_args()

input_dict_directory = args.input_dict_directory
output_train_dictionary = args.output_train_dict
output_tune_dictionary = args.output_tune_dict
output_test_dictionary = args.output_test_dict
cohort_split = args.cohort_split
train_cohorts = str.split(args.train_cohorts, ",")
test_cohorts = str.split(args.test_cohorts, ",")
tune_proportion = args.tune_proportion
test_proportion = args.test_proportion
if args.exclude_list:
    exclude = pd.read_csv(args.exclude_list)
    exclude_list = exclude["files"].tolist()
    print(f"Exclusion list: {args.exclude_list} length: {len(exclude_list)}")
    exclude_list = [x + "_pdac_dict.pth" for x in exclude_list]
else:
    exclude_list = None

np.random.seed(416)


def get_dicts(path):
    """
    Input:
    path: directory with dictionaries

    Output:
    list of the paths
    """
    dicts = []
    for root, _, files in os.walk(path):
        files.sort()
        for name in files:
            dicts_path = os.path.join(root, name)
            if name.endswith(".pth"):
                dicts.append(dicts_path)
    return dicts


def combine_dictionaries(input_dir, exclude):
    """
    Input:
    input_dir: A directory with .pth files to be combined the .pth files must have keys time and vital
    exclude: A list of filenames to exclude

    Output:
    full_dictionary: A combined dictionary without the excluded files
    """

    dictionary_paths = get_dicts(input_dir)
    if not exclude:
        exclude = []
    full_dictionary = defaultdict(list)
    slide_n = 0
    excluded_n = 0
    for i, filename in enumerate(dictionary_paths):
        slide_n += 1
        if filename.split("/")[-1] in exclude:
            # print('Excluding ', filename.split('/')[-1])
            excluded_n += 1
        else:  # append the torch dict to the full dictionary if not in exclusion list
            loaded_dictionary = torch.load(dictionary_paths[i])
            for key, value in loaded_dictionary.items():
                full_dictionary[key].append(value)

    print("{} slides".format(slide_n))
    print("{} excluded on review".format(excluded_n))
    return full_dictionary


def list_duplicate_indices(full_list):
    """
    Returns a list of indices for duplicated elements within a list. Used to exclude multiple slides from the same patient.
    """
    unduplicated_set = set()
    duplicate_idx = []
    for idx, val in enumerate(full_list):
        if val not in unduplicated_set:
            unduplicated_set.add(val)
        else:
            # print(f'{val} is duplicated, exclude')
            duplicate_idx.append(idx)
    return duplicate_idx


def split_dictionary(full_dictionary, tune_proportion, test_proportion):
    """
    Split dictionary into training, tuning, and testing sets at the slide level
    and randomly shuffle.
    """
    train_dictionary = {}
    tune_dictionary = {}
    test_dictionary = {}

    # idx_full based on the number of slide dicts in the input dict directory
    idx_full = np.arange(len(full_dictionary["slide_id"]))
    # duplicate indices based on patient ids from the slide dicts in the input dict directory
    duplicate_indices = list_duplicate_indices(full_dictionary["patient_id"])
    print("{} duplicate slides".format(len(duplicate_indices)))

    if not cohort_split:
        print(f"No cohort split, choosing idxs from {train_cohorts}")

        train_idx = []
        tune_idx = []
        test_idx = []

        # sample the same proportion of indices from each cohort for train/tune
        for cohort in train_cohorts:
            # get the indices that match the train cohort
            idx = idx_full[np.isin(full_dictionary["cohort"], [cohort])]
            # remove the indices with duplicated patients
            idx = np.setdiff1d(idx, duplicate_indices).astype(np.intp)

            # train
            print("TRAIN SHAPE", idx.shape[0])
            train = list(
                np.random.choice(
                    idx,
                    np.ceil(
                        (1 - tune_proportion - test_proportion) * idx.shape[0]
                    ).astype(np.intp),
                    replace=False,
                )
            )
            train_idx.extend(train)

            # tune
            other_idx = np.setdiff1d(idx, train).astype(np.intp)
            tune = list(
                np.random.choice(
                    other_idx,
                    np.ceil(
                        (tune_proportion / (tune_proportion + test_proportion))
                        * other_idx.shape[0]
                    ).astype(np.intp),
                    replace=False,
                )
            )
            tune_idx.extend(tune)

            # test
            test = list(np.setdiff1d(other_idx, tune).astype(np.intp))
            test_idx.extend(test)

        random.shuffle(train_idx)
        random.shuffle(tune_idx)
        random.shuffle(test_idx)

        print(f"Train idx: {train_idx}")
        print(f"Tune idx: {tune_idx}")
        print(f"Test idx: {test_idx}")

        # Keys are magnification, cohort, slide_id, patient_id, grid, etc.
        for key in full_dictionary.keys():
            train_dictionary[key] = np.array(full_dictionary[key])[train_idx].tolist()
            tune_dictionary[key] = np.array(full_dictionary[key])[tune_idx].tolist()
            test_dictionary[key] = np.array(full_dictionary[key])[test_idx].tolist()
        return train_dictionary, tune_dictionary, test_dictionary

    else:  # Cohort split, train cohorts are split into train/tune, test cohorts remain in test set

        train_idx = []
        tune_idx = []

        # sample the same proportion of indices from each cohort for train/tune
        for cohort in train_cohorts:
            # get the indices that match the train cohort
            idx = idx_full[np.isin(full_dictionary["cohort"], [cohort])]
            # remove the indices with duplicated patients
            idx = np.setdiff1d(idx, duplicate_indices).astype(np.intp)

            # tune
            tune = list(
                np.random.choice(
                    idx,
                    np.ceil(tune_proportion * idx.shape[0]).astype(np.intp),
                    replace=False,
                )
            )
            tune_idx.extend(tune)

            # train
            train = list(np.setdiff1d(idx, tune).astype(np.intp))
            train_idx.extend(train)

        random.shuffle(train_idx)
        random.shuffle(tune_idx)

        # test
        test_idx = idx_full[np.isin(full_dictionary["cohort"], test_cohorts)]
        test_idx = np.setdiff1d(test_idx, duplicate_indices).astype(np.intp)
        np.random.shuffle(test_idx)

        print(f"Train idx: {train_idx}")
        print(f"Tune idx: {tune_idx}")
        print(f"Test idx: {test_idx}")

        # Keys are magnification, cohort, slide_id, patient_id, grid, etc.
        for key in full_dictionary.keys():
            train_dictionary[key] = np.array(full_dictionary[key])[train_idx].tolist()
            tune_dictionary[key] = np.array(full_dictionary[key])[tune_idx].tolist()
            test_dictionary[key] = np.array(full_dictionary[key])[test_idx].tolist()

        return train_dictionary, tune_dictionary, test_dictionary


def main():
    full_dictionary = combine_dictionaries(input_dict_directory, exclude_list)
    train_dictionary, tune_dictionary, test_dictionary = split_dictionary(
        full_dictionary, tune_proportion, test_proportion
    )
    print(
        "Train: {}, tune: {}, test: {}".format(
            len(train_dictionary["patient_id"]),
            len(tune_dictionary["patient_id"]),
            len(test_dictionary["patient_id"]),
        )
    )
    torch.save(train_dictionary, output_train_dictionary)
    torch.save(tune_dictionary, output_tune_dictionary)
    torch.save(test_dictionary, output_test_dictionary)


if __name__ == "__main__":
    main()
