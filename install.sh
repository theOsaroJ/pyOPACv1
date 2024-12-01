## make a mininconda if not already avaialable
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

## software depends heavily on openbabel ##
conda create -n openbabel_env
conda activate openbabel_env
conda install conda-forge::openbabel

pip install -r requirements.txt

pip install -e .
