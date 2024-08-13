#!/bin/bash

set -e

SEQ_LEN=72

MASK_NUM_ITERATIONS=8

for i in $(seq 1 $MASK_NUM_ITERATIONS)
do
  # set mask missing percent
  miss_percent=$((10*i))

  echo ""
  echo "-----------------------------------------------------------------------"
  echo "| Mask missing rate: $miss_percent "
  echo "-----------------------------------------------------------------------"
  echo ""

  # generate dataset
  python gene_DSM2_dataset.py \
    --file_path DSM2_data \
    --mask_type sparse \
    --seq_len $SEQ_LEN \
    --miss_percent $miss_percent \
    --dataset_name DSM2_seqlen${SEQ_LEN}_sparse_miss${miss_percent} \
    --saving_path ../generated_datasets

done
