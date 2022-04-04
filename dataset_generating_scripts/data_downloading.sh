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
cat Outcomes-a.txt > mega_outcomes.txt
cat Outcomes-b.txt >> mega_outcomes.txt
cat Outcomes-c.txt >> mega_outcomes.txt

# for Air-Quality
cd .. && mkdir AirQuality && cd AirQuality
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip
unzip PRSA2017_Data_20130301-20170228.zip

# for Electricity
cd .. && mkdir Electricity && cd Electricity
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip
