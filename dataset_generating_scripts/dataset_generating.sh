# generate physio2012 dataset
python gene_PhysioNet2012_dataset.py \
  --raw_data_path RawData/Physio2012_mega/mega \
  --outcome_files_dir RawData/Physio2012_mega/ \
  --dataset_name physio2012_37feats_01masked \
  --saving_path ../generated_datasets

# generate UCI Beijing air quality dataset
python gene_UCI_BeijingAirQuality_dataset.py \
  --file_path RawData/AirQuality/PRSA_Data_20130301-20170228 \
  --seq_len 24 \
  --artificial_missing_rate 0.1 \
  --dataset_name AirQuality_seqlen24_01masked \
  --saving_path ../generated_datasets

# generate UCI electricity dataset
python gene_UCI_electricity_dataset.py \
  --file_path RawData/Electricity/LD2011_2014.txt \
  --artificial_missing_rate 0.1 \
  --seq_len 100 \
  --dataset_name Electricity_seqlen100_01masked \
  --saving_path ../generated_datasets

