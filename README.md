# torchmd
<p align="left">
  <img src="assets/logo.jpg" width="200">
</p> 

PyTorch code for End-to-end differetiable molecular simulations. Complete code and demo coming soon.

Paper: 

Wang, W., Axelrod, S., & Gómez-Bombarelli, R. (2020). Differentiable Molecular Simulations for Control and Learning. ArXiv. Retrieved from https://arxiv.org/abs/2003.00868

<p align="center">
  <img src="assets/schematic.jpg" width="400">
</p>

### Applications

#### End-to-End Fitting for Macroscopic/Coarse-Grained Observable 
Backpropagating through the trajectory to train a GNN that reproduces a target pair distribution function.
We demonstrated the fitting of water rdf (Oxygen-Oxygen) at 298k with differentiable simulations
<p align="center">
  <img src="assets/water_gnn_rdf_298k.gif" width="300">
</p>


#### Controllable Fold for polymer chain 
Folding a polymer with Graph Neural Networks 

<p align="center">
  <img src="assets/fold.gif" width="300">
</p>


#### Quantum Isomerization 

We fit electric field to optimize efficiency of a quantum isomerization process [Coming]



