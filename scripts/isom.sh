cwd=$(pwd)
torchmd_path=${cwd//'scripts'/''}
export PYTHONPATH=$torchmd_path:$PYTHONPATH

source activate nff
python isom.py  -logdir isom -lr 1e-4 -device 0 -nepochs 40

