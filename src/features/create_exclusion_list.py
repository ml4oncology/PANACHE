import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--icgc_file", type=str, help="ICGC clinical file")
parser.add_argument("--tcga_file", type=str, help="TCGA file with exclusion list")
parser.add_argument(
    "--out_file",
    default=None,
    type=str,
    help="Output pandas file with patient_id field to exclude on",
)
args = parser.parse_args()

icgc = pd.read_csv(args.icgc_file)
exclude_icgc = icgc[(icgc["type"].isin(["FS"])) | (icgc["duplicate"] == 1)]["file"]
exclude_icgc = exclude_icgc.values.tolist()
exclude_icgc = [x.split(".")[0] for x in exclude_icgc]

tcga = pd.read_csv(args.tcga_file, header=None, names=("slide", "review", "other"))
exclude_tcga = tcga[~tcga["review"].isin(["Include"])]["slide"]
exclude_tcga = exclude_tcga.values.tolist()
exclude_tcga = [x[:16] for x in exclude_tcga]

exclude = exclude_icgc + exclude_tcga

for x in exclude:
    print(x)

if args.out_file is not None:
    df = pd.DataFrame(data={"files": exclude})
    df.to_csv(args.out_file, sep=",", index=False)
    print("{} files excluded and written to {}".format(len(exclude), args.out_file))
