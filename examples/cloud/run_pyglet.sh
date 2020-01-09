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



# apt-get install freeglut3-dev --yes
conda create -n saferew python=3.6 -y
source activate saferew
pip install pyglet moviepy
# #python -c "from pyglet.gl import *"
# python -c "import pyglet"
# python -c "from pyglet import gl, graphics"
python -c "from moviepy.editor import ImageSequenceClip
"
# python -c "import os; print(os.environ['DISPLAY'])"
