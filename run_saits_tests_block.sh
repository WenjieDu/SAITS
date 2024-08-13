#!/bin/bash

set -e

SEQ_LEN=72
SAITS_MODEL="DSM2_SAITS"

MASK_FEATURES_NUM=4
NUM_ITERATIONS=36

# create results file
timestamp=$(date +%s)
results_file_path="saits_test_results/tests_results_${SAITS_MODEL}_block_maskfeat${MASK_FEATURES_NUM}_${timestamp}.csv"
touch $results_file_path
printf "Run,Miss_len,MAE,RMSE,MRE\n" >> $results_file_path

# run multiple iterations with different missing block sizes
for i in $(seq 1 $NUM_ITERATIONS)
do
    mask_block_len=$((2*i))

    echo ""
    echo "-------------------------------------------------------------------"
    echo "| Test run $i | Mask num. features: $MASK_FEATURES_NUM Mask block length: $mask_block_len"
    echo "-------------------------------------------------------------------"

    # select dataset
    printf "\n---- Selecting dataset ----\n\n"
    cd generated_datasets
    if [ -d DSM2 ]; then
        rm -rf DSM2
    fi
    dataset_path=DSM2_seqlen${SEQ_LEN}_block_maskfeat${MASK_FEATURES_NUM}_blocklen${mask_block_len}
    ln -s $dataset_path DSM2
    cd ..
    echo "selected $dataset_path"

    # run SAITS model
    printf "\n---- Running SAITS model ----\n\n"
    python run_models.py --config_path configs/${SAITS_MODEL}.ini --test_mode

    # get test metrics
    test_log_file=$(ls NIPS_results/${SAITS_MODEL} | tail -n 1)
    mae=$(grep -i 'imputation_MAE'  NIPS_results/${SAITS_MODEL}/$test_log_file | awk 'NF{ print $NF }')
    rmse=$(grep -i 'imputation_RMSE'  NIPS_results/${SAITS_MODEL}/$test_log_file | awk 'NF{ print $NF }')
    mre=$(grep -i 'imputation_MRE'  NIPS_results/${SAITS_MODEL}/$test_log_file | awk 'NF{ print $NF }')

    # save test metrics to file
    printf "\n---- Saving results ----\n\n"
    printf "%i,%i,%4.3f,%4.3f,%4.3f\n" $i $mask_block_len $mae $rmse $mre >> $results_file_path
    echo "saved results to $results_file_path"

done
