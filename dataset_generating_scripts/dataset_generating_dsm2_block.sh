#!/bin/bash

set -e

SEQ_LEN=72

MASK_NUM_FEAT_ITERATION=4
MASK_NUM_BLK_ITERATIONS=36

for i in $(seq 1 $MASK_NUM_FEAT_ITERATION)
do
  for j in $(seq 1 $MASK_NUM_BLK_ITERATIONS)
  do
    # set masked block length
    mask_block_len=$((2*j))

    echo ""
    echo "-----------------------------------------------------------------------"
    echo "| Mask num. features: $i , block length: $mask_block_len"
    echo "-----------------------------------------------------------------------"
    echo ""

    # generate dataset
    python gene_DSM2_dataset.py \
      --file_path DSM2_data \
      --mask_type block \
      --seq_len $SEQ_LEN \
      --mask_features_num $i \
      --mask_block_len $mask_block_len \
      --dataset_name DSM2_seqlen${SEQ_LEN}_block_maskfeat${i}_blocklen${mask_block_len} \
      --saving_path ../generated_datasets

  done
done