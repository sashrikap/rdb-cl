## Instal conda
apt-get update --yes
apt-get upgrade --yes
apt-get install wget freeglut3-dev --yes
apt-get install xvfb --yes
apt-get install mesa-utils --yes
apt-get install libfontconfig1-dev pkg-config --yes
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
pip install imageio-ffmpeg

## Set up input folder
# wget https://storage.googleapis.com/active-ird-experiments/rss-logs/logs/input/200110_test_eval_all.tar.gz
# tar xvzf 200110_test_eval_all.tar.gz
# mkdir /gcp_input
# mv 200110_test_eval_all /gcp_input/

## Run experiment
# DISPLAY=':99.0'
# Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:0


#python ./rdb/examples/run_interactive.py --GCP_MODE
python ./rdb/examples/tune/make_mp4.py --GCP_MODE
