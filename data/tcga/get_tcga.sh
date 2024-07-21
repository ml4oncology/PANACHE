#!/bin/bash

# Download TCGA data
root=$(pwd)
cd $root/data/tcga/images_raw
$root/data/tcga/gdc-client download -m $root/data/tcga/gdc_manifest.txt