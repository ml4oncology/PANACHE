# input arguments
wsi_dir=$1
mag=$2
jpg_dir=$3
dict_dir=$4
ftr_dir=$5
output_csv_filepath=$6

# save wsi dicts and jpgs
# assuming files in format <slide_id>.svs
python ./src/model/HIPT/inference/get_wsi_tiles_dicts.py --input_slide_dir ${wsi_dir} --magnification ${mag} --output_jpg_dir ${jpg_dir} --output_dict_dir ${dict_dir}

# save 4k features, requires gpu(s)
python ./src/model/HIPT/inference/save_4k_ftrs.py --jpg_dir ${jpg_dir} --wsi_dict ${dict_dir}/wsi_coord_dict.pth --output_ftr_dir ${ftr_dir} --path_to_checkpoints ./src/model/HIPT/pretrain_checkpoints

# save model predictions
python ./src/model/HIPT/inference/get_model_predictions.py --wsi_dict ${dict_dir}/wsi_coord_dict.pth --ftr_directory ${ftr_dir} --checkpoint_file ./src/model/HIPT/inference/mean_pooling_best_model.pt --output_csv ${output_csv_filepath}