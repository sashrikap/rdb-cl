## Instal conda
apt-get update --yes
apt-get upgrade --yes
apt-get install wget --yes
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh
PATH="/root/miniconda/bin:$PATH"
apt-get install python3-pip --yes
pip install --upgrade pip

## create env
conda create -n saferew python=3.6 -y
source activate saferew
pip install -e ./rdb

## Run experiment
#python ./rdb/examples/run_acquisition.py
python ./rdb/examples/run_highway.py
