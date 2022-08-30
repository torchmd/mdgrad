# learning pair potential experiments 

## LJ systems 

### fit for rho = 0.3
```python run_lj.py -nruns 1 -device 0 -logdir lj_fit_0.3_0.5_0.7 -name lj -sigma 0.9 -update_freq 1 -cutoff 2.5 -dt 0.005 -vacf_weight 0.0 -data lj_0.3_1.2 lj_0.5_1.2 lj_0.7_1.2```

## mixture systems 
### training for x = 0.25 and test for x = 0.5, 0.75
```python fit_mix.py -logdir trainx_0.25 -subjob test -device 0 -trainx 0.25 -valx 0.5 0.75 -nepochs 200 -nsim 100 -sigma 1.0 -n_layers 2  -n_width 105 -nonlinear LeakyReLU -gaussian_width 0.15 -lr 5e-3 -nruns 1 --res -update_epoch 5```

### simultanoues training for x = 0.25, 0.5, 0.75
```python fit_mix.py -logdir trainx_0.75_0.5_0.25 -subjob test -device 0 -trainx 0.75 0.5 0.25  -nepochs 200 -nsim 100 -sigma 1.0 -n_layers 2  -n_width 105 -nonlinear LeakyReLU -gaussian_width 0.15 -lr 5e-3 -nruns 1 --res -update_epoch 5```


## CG water 
### simultanoues training for T=288K, 338K, 388K
```python run_water.py -logdir temp_dependent_water_train_all -data H20_288K_spce H20_338K_spce H20_388K_spce -name water -device 2 --tpair -nruns 1 -nepochs 600 -nsim 500'''


