#!/bin/sh

mkdir logging

# see https://github.com/flatironinstitute/disBatch
# for launch instructions
batch_estimate_disbatch.py \
    --biom-table data/table.biom \
    --metadata-file data/metadata.txt \
    --replicates reps \
    --batches batch \
    --local-directory logging \
    --job-extra 'module load disBatch/2.0-beta' \
    --output-inference output.nc

rm tasks*
rm slurm*
rm output.nc
