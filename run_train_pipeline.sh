# Create list of patients to exclude (based on pathologist review)
python ./src/features/create_exclusion_list.py --tcga_file ./data/tcga/tcga_review.csv --icgc_file ./data/icgc/icgc_clinical.csv --out_file ./data/exclude_files.csv

#Save 4096x4096 WSI tiles as jpgs and dictionaries with WSI metadata
# TCGA
python ./src/features/process_slides.py \
    --input_svs_directory ./data/tcga/images_raw \
    --clinical_file ./data/tcga/nationwidechildrens.org_clinical_patient_paad.txt \
    --cohort tcga \
    --output_dict_directory ./data/slide_dicts \
    --output_jpg_directory ./data/tile_jpgs_4k \
    --magnification 40 \
    --log_dir ./logs \
    --size 4096 \
    --tile_tissue_proportion 0.2
#CPTAC
python ./src/features/process_slides.py \
    --input_svs_directory ./data/cia/images_raw \
    --clinical_file ./data/cia/cia_clinical.csv \
    --cohort cia \
    --output_dict_directory ./data/slide_dicts \
    --output_jpg_directory ./data/tile_jpgs_4k \
    --magnification 20 \
    --log_dir ./logs \
    --size 4096 \
    --tile_tissue_proportion 0.2
#ICGC
# python ./src/features/process_slides.py \
#     --input_svs_directory ./data/icgc/images_raw \
#     --clinical_file ./data/icgc/icgc_clinical.csv \
#     --cohort icgc \
#     --output_dict_directory ./data/slide_dicts \
#     --output_jpg_directory ./data/tile_jpgs_4k \
#     --magnification 20 \
#     --log_dir ./logs \
#     --size 4096 \
#     --tile_tissue_proportion 0.2

# Create dictionary files for train, validation, and test
mkdir -p ./data/slide_dicts
python ./src/features/merge_and_split_dicts.py \
    --input_dict_directory ./data/slide_dicts \
    --output_train_dict ./data/split_dicts/train_dict.pth \
    --output_tune_dict ./data/split_dicts/tune_dict.pth \
    --output_test_dict ./data/split_dicts/test_dict.pth \
    --exclude_list ./data/exclude_files.csv \
    --cohort_split \
    --train_cohorts cia,tcga

# Save HIPT features from ViT 4k from saved dictionary files
python ./src/model/HIPT/save_ftrs_hipt_4k.py --jpg_dir ./data/tile_jpgs_4k --ftr_dir ./data/saved_ftrs_hipt_4k --lib ./data/split_dicts/clean_samples/train_dict.pth
python ./src/model/HIPT/save_ftrs_hipt_4k.py --jpg_dir ./data/tile_jpgs_4k --ftr_dir ./data/saved_ftrs_hipt_4k --lib ./data/split_dicts/clean_samples/tune_dict.pth
python ./src/model/HIPT/save_ftrs_hipt_4k.py --jpg_dir ./data/tile_jpgs_4k --ftr_dir ./data/saved_ftrs_hipt_4k --lib ./data/split_dicts/clean_samples/test_dict.pth

# Model training with bayes optimization
python ./src/model/HIPT/train_tiles_vit4096_mean_pool_bayes_optim.py --train_lib ./data/split_dicts/clean_samples/train_dict.pth --tune_lib ./data/split_dicts/clean_samples/tune_dict.pth --ftr_directory ./data/saved_ftrs_hipt_4k --workers 0 --nepochs 300 --runs_csv_location ./src/model/HIPT/runs --save_model_location ./src/model/HIPT/saved_models --max_epochs_no_improvement 10 --bayes_sweep_n 150