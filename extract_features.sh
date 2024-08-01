

CUDA_VISIBLE_DEVICES=0 python extract_features_simclr_TCGA.py --data_h5_dir 'path after create patches'
--data_slide_dir "path ori slide" --csv_path dataset_csv/TCGA_label.csv \
--feat_dir "result dir" --batch_size 64 --slide_ext .svs

