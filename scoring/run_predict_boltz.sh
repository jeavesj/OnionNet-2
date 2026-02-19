#!/bin/bash



scaler_fpath=../models/train_scaler.scaler
model_fpath=../models/62shell_saved-model.h5
dir_list=$(ls /mnt/scratch/jeaves/CASF-2016/boltz2)
# dir_list=$(ls /mnt/scratch/jeaves/OnionNet-2/samples)

for item in $dir_list; do
    rec_fpath=/mnt/scratch/jeaves/OnionNet-2/boltz2/${item}/${item}_boltz2_model_0_protein.pdb
    lig_pdb_fpath=/mnt/scratch/jeaves/OnionNet-2/boltz2/${item}/${item}_boltz2_model_0_ligand.pdb
    lig_sdf_fpath=/mnt/scratch/jeaves/OnionNet-2/boltz2/${item}/${item}_boltz2_model_0_ligand.sdf
    out_fpath=/mnt/scratch/jeaves/OnionNet-2/boltz2/${item}/pred.csv

    if [ ! -f "$ligand_pdb_fpath" ]; then
        obabel "$lig_sdf_fpath" -O "$lig_pdb_fpath"
    fi

    python predict.py \
        -rec_fpath $rec_fpath \
        -lig_fpath $lig_pdb_fpath \
        -shape 84,124,1 \
        -scaler $scaler_fpath \
        -model $model_fpath \
        -shells 62 \
        -out_fpath $out_fpath
done

