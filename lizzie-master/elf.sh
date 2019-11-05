#!/bin/bash
source activate dnn
cd ~/ELF/scripts

export ELF_DEVELOPMENT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
export PYTHONPATH="${ELF_DEVELOPMENT_ROOT}"/src_py/:"${ELF_DEVELOPMENT_ROOT}"/build/elf/:"${ELF_DEVELOPMENT_ROOT}"/build/elfgames/go/:${PYTHONPATH}

cd ~/ELF/scripts/elfgames/go

game=elfgames.go.game model=df_pred model_file=elfgames.go.df_model3 python3 df_console.py --mode online --keys_in_reply V rv \
    --use_mcts --mcts_verbose_time --mcts_use_prior --mcts_persistent_tree --load ./pretrained-go-19x19-v1.bin \
    --server_addr localhost --port 1234 \
    --replace_prefix resnet.module,resnet \
    --no_check_loaded_options \
    --no_parameter_print \
    --verbose --gpu 0 --num_block 20 --dim 224 --mcts_puct 1.50 --batchsize 16 --mcts_rollout_per_batch 16 --mcts_threads 2 --mcts_rollout_per_thread 200 --resign_thres 0.05 --mcts_virtual_loss 1


