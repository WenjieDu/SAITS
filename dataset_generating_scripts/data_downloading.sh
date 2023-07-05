#  The shell script to download used datasets.
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


mkdir RawData && cd RawData

# for PhysioNet-2012
mkdir Physio2012_mega && cd Physio2012_mega
wget https://www.physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download -O set-a.tar.gz
wget https://www.physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download -O set-b.tar.gz
wget https://www.physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download -O set-c.tar.gz

wget https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download -O Outcomes-a.txt
wget https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download -O Outcomes-b.txt
wget https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download -O Outcomes-c.txt

tar -zxf set-a.tar.gz && tar -zxf set-b.tar.gz && tar -zxf set-c.tar.gz
mkdir mega && mv set-a/* mega && mv set-b/* mega && mv set-c/* mega

# for Air-Quality
cd .. && mkdir AirQuality && cd AirQuality
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip
unzip PRSA2017_Data_20130301-20170228.zip

# for Electricity
cd .. && mkdir Electricity && cd Electricity
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip

# for Electricity Transformer Temperature (ETT)
cd .. && mkdir ETT && cd ETT
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv