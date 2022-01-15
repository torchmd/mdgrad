```
you first need to install 

```
pip install sigopt 
pip install matplotlib
pip install torch
pip install scipy
```

To run a LJ fitting example: 
```
python run_lj.py -nruns 1 -logdir test_lj_0.7_0.4 -name lj -sigma 0.9 -update_freq 1 -cutoff 2.5 -dt 0.005 -vacf_weight 0.0 -data lj_0.7_0.4
```