#  The shell script to run dataset-generating python scripts.
#
#  If you use code in this repository, please cite our paper as below. Many thanks.
#
#  @article{DU2023SAITS,
#  title = {{SAITS: Self-Attention-based Imputation for Time Series}},
#  journal = {Expert Systems with Applications},
#  volume = {219},
#  pages = {119619},
#  year = {2023},
#  issn = {0957-4174},
#  doi = {https://doi.org/10.1016/j.eswa.2023.119619},
#  url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
#  author = {Wenjie Du and David Cote and Yan Liu},
#  }
#
#  or
#
#  Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619


# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


# generate physio2012 dataset
python gene_PhysioNet2012_dataset.py \
  --raw_data_path RawData/Physio2012_mega/mega \
  --outcome_files_dir RawData/Physio2012_mega/ \
  --dataset_name physio2012_37feats_01masked_1 \
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

