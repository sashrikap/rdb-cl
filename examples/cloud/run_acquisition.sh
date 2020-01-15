## Instal conda
apt-get update --yes
apt-get upgrade --yes
apt-get install wget freeglut3-dev --yes
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh

PATH="/root/miniconda/bin:$PATH"
DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

apt-get install python3-pip --yes
pip install --upgrade pip

## create env
conda create -n saferew python=3.6 -y
source activate saferew
pip install -e ./rdb

## Set up input folder
#wget https://www.dropbox.com/s/a0spfbfrl557fe9/200110_test_eval_all.tar.gz
wget https://storage.googleapis.com/active-ird-experiments/rss-logs/logs/input/200110_test_eval_all.tar.gz
tar xvzf 200110_test_eval_all.tar.gz
mkdir /gcp_input
mv 200110_test_eval_all /gcp_input/

## Run experiment
# touch /gcp_output/test.txt
# python ./rdb/examples/cloud/run_filetest.py
# python ./rdb/examples/run_highway.py
# python ./rdb/examples/run_acquisition.py
python ./rdb/examples/run_acquisition.py --GCP_MODE
