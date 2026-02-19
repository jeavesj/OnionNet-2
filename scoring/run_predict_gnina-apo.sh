#!/bin/bash --login
#SBATCH --job-name=onion-gnina-apo
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-100%20
#SBATCH --output=/mnt/scratch/jeaves/OnionNet-2/logs/%x_%A_%a.out
#SBATCH --error=/mnt/scratch/jeaves/OnionNet-2/logs/%x_%A_%a.err

set -euo pipefail

module purge
module load Miniforge3
conda activate onionnet2_inf_py38

SRC_BASE_DIR='/mnt/scratch/jeaves/CASF-2016'
DEST_BASE_DIR='/mnt/scratch/jeaves/OnionNet-2'
MAIN_DIR='/mnt/research/woldring_lab/Members/Eaves/FAIR_PLBAP/results/gnina-apo'

scaler_fpath='/mnt/scratch/jeaves/OnionNet-2/models/train_scaler.scaler'
model_fpath='/mnt/scratch/jeaves/OnionNet-2/models/62shell_saved-model.h5'

rank_tag="$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")"
SRC_DIR="${SRC_BASE_DIR}/gnina-apo_best${rank_tag}"
IN_DIR="${DEST_BASE_DIR}/gnina-apo_best${rank_tag}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_CSV="${DEST_BASE_DIR}/CASF2016_gnina-apo-best${rank_tag}_onionnet2_preds_and_times.csv"

echo "=== Rank ${rank_tag} ==="
echo "Source: ${SRC_DIR}"
echo "Work dir: ${IN_DIR}"

if [ ! -d "${SRC_DIR}" ]; then
    echo "[ERROR] Missing SRC_DIR: ${SRC_DIR}"
    exit 1
fi

rm -rf "${IN_DIR}"
cp -r "${SRC_DIR}" "${IN_DIR}"

shopt -s nullglob

processed=0
skipped_missing_rec=0
skipped_missing_lig=0
failed_pred=0

for item_dir in "${IN_DIR}"/*; do
    [ -d "${item_dir}" ] || continue
    item="$(basename "${item_dir}")"

    rec_fpath="${item_dir}/${item}_protein.pdb"
    lig_pdb_fpath="${item_dir}/${item}_ligand.pdb"
    lig_sdf_fpath="${item_dir}/${item}_ligand.sdf"
    out_fpath="${item_dir}/pred.csv"

    echo "[INFO] Processing ${item}"
    echo "       rec: ${rec_fpath}"
    echo "       lig pdb: ${lig_pdb_fpath}"
    echo "       lig sdf: ${lig_sdf_fpath}"

    if [ ! -f "${rec_fpath}" ]; then
        echo "[WARN] Missing receptor: ${rec_fpath} (skipping ${item})"
        skipped_missing_rec=$((skipped_missing_rec + 1))
        continue
    fi

    if [ ! -f "${lig_pdb_fpath}" ]; then
        if [ -f "${lig_sdf_fpath}" ]; then
            echo "[INFO] Converting SDF to PDB for ${item}"
            obabel "${lig_sdf_fpath}" -O "${lig_pdb_fpath}"
        else
            echo "[WARN] Missing ligand pdb and sdf for ${item} (skipping)"
            skipped_missing_lig=$((skipped_missing_lig + 1))
            continue
        fi
    fi

    rm -f "${out_fpath}"

    if ! python "${DEST_BASE_DIR}/scoring/predict.py" \
        -rec_fpath "${rec_fpath}" \
        -lig_fpath "${lig_pdb_fpath}" \
        -shape 84,124,1 \
        -scaler "${scaler_fpath}" \
        -model "${model_fpath}" \
        -shells 62 \
        -out_fpath "${out_fpath}" \
        > "${item_dir}/predict.log" 2>&1
    then
        echo "[ERROR] predict.py failed for ${item} (see ${item_dir}/predict.log)"
        failed_pred=$((failed_pred + 1))
        continue
    fi

    if [ ! -f "${out_fpath}" ]; then
        echo "[ERROR] predict.py returned success but no pred.csv for ${item}"
        failed_pred=$((failed_pred + 1))
        continue
    fi

    processed=$((processed + 1))
done

n_dirs=$(find "${IN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_preds=$(find "${IN_DIR}" -mindepth 2 -maxdepth 2 -name 'pred.csv' | wc -l)

echo "[INFO] Rank ${rank_tag}: complexes=${n_dirs}, pred.csv files=${n_preds}"
echo "[INFO] Rank ${rank_tag}: processed=${processed}, skipped_missing_rec=${skipped_missing_rec}, skipped_missing_lig=${skipped_missing_lig}, failed_pred=${failed_pred}"

python3 "${DEST_BASE_DIR}/merge_predictions.py" \
    --basedir "${IN_DIR}" \
    --filename pred.csv \
    --output "${OUT_CSV}"

mkdir -p "${MAIN_DIR}"
cp "${OUT_CSV}" "${MAIN_DIR}/"

rm -rf "${IN_DIR}"
echo "[DONE] Wrote ${OUT_CSV}"
