    
from torchmd.system import generate_nbr_list 
from nff.data import Dataset, split_train_validation_test, collate_dicts, sparsify_tensor
from nff.train import Trainer, get_trainer, get_model, loss, hooks, metrics, evaluate
import nff.data as d
from torch.utils.data import DataLoader
import numpy as np
import torch
from ase.geometry import wrap_positions


def pretrain_aimd(model, atoms, device, cutoff, path, n_epochs):

    size = 4 
    n_mols = size ** 3
    z = atoms.get_atomic_numbers()

    f_data = [np.load('../data/water_aimd/force_{}.npy'.format(i)).reshape(-1, n_mols * 3, 3) for i in range(4)]
    xyz_data = [np.load('../data/water_aimd/coord_{}.npy'.format(i)).reshape(-1, n_mols * 3, 3) for i in range(4)]
    box = np.load('../data/water_aimd/box_0.npy').reshape(-1,  3, 3)

    f_data = np.concatenate(f_data)
    xyz_data = np.concatenate(xyz_data)

    props = {'nxyz': [],
             'offsets': [],
             'nbr_list': [], 
             'cell': [],
             'energy_grad': []}
    
    print("building dataset") 
    for i, force in enumerate(f_data):
        f_dft = torch.Tensor(force).to(device)
        coord = wrap_positions(xyz_data[i], cell=box[0])

        nxyz = np.concatenate([
                np.array(z).reshape(-1, 1),
                coord
            ], axis=1)

        nbr_list, offsets = generate_nbr_list(torch.Tensor(coord), cutoff, cell=torch.Tensor(np.diag(box[0])))
        offsets = offsets[nbr_list[:,0], nbr_list[:,1], :]

        props['nxyz'].append(torch.Tensor(nxyz))
        props['nbr_list'].append(nbr_list)
        props['offsets'].append(offsets)
        props['cell'].append(torch.Tensor(box[0]))
        props['energy_grad'].append(torch.Tensor(-force))
        
    props['offsets'] = [sparsify_tensor(offset.matmul(torch.Tensor(props["cell"][i])))
                    for i, offset in enumerate(props['offsets'])]
        
    dataset = d.Dataset(props.copy(), units='kcal/mol')
    
    train, val, test = split_train_validation_test(dataset, val_size=0.05, test_size=0.05)
    
    train_loader = DataLoader(train, batch_size=4, collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=4, collate_fn=collate_dicts)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4)


    train_metrics = [
        metrics.MeanAbsoluteError('energy_grad')
    ]

    from shutil import rmtree
    import os

    loss_fn = loss.build_mse_loss(loss_coef={'energy_grad': 1})

    OUTDIR = path + '/model/'
    train_hooks = [
        hooks.MaxEpochHook(1000),
        hooks.CSVHook(
            OUTDIR,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(
            OUTDIR,
            metrics=train_metrics,
            separator = ' | ',
            time_strf='%M:%S'
        ),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=30,
            factor=0.5,
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True
        )
    ]

    T = Trainer(
        model_path=OUTDIR,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks
    )
    
    T.train(device=device, n_epochs=n_epochs)
    
    return T.get_best_model()