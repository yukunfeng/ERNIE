set -x

source ./run.sh

function run_tacred_with_warmuphyper_e() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
    local warmup=$7
    local lr=$8
    local epoch=$9
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}warm${warmup}lr${lr}e${epoch}"
    
    python3 code/run_tacred_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/tacred"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   \
        --learning_rate $lr   --num_train_epochs $epoch \
        --output_dir "tmp"   \
        --loss_scale 128 --entities_tsv $entities_tsv  --warmup_proportion $warmup
}
layer=-2
# Run tacred
data="data/tacred"
emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base

# run_tacred 100 short 100 1 $layer $emb_base

# run_tacred 100 short 0 5 $layer $emb_base
# run_tacred 100 short 0 1 $layer $emb_base

# run_tacred 100 short 0.3 5 $layer $emb_base
# run_tacred 100 short 0.3 1 $layer $emb_base

# run_tacred 0.3 short 0.3 5 $layer $emb_base
# run_tacred 0.3 short 0.3 1 $layer $emb_base


run_tacred_with_warmuphyper 0.5 short 0.5 5 $layer $emb_base 0.1 4e-5
# run_tacred_with_warmuphyper 0.3 short 0.3 5 $layer $emb_base 0.1 4e-5
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.05 4e-5 4
# run_tacred_with_warmuphyper_e 100 short 100 5 $layer $emb_base 0.1 3e-5 4
# run_tacred_with_warmuphyper_e 100 short 0.3 5 $layer $emb_base 0.1 3e-5 4
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.1 3e-5 4
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.1 3e-5 5
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.1 5e-5 4
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.1 3e-5 6
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.05 3e-5 4
# run_tacred_with_warmuphyper_e 0.3 short 0.3 5 $layer $emb_base 0.1 4e-5 6
# run_tacred_with_warmuphyper 0.3 short 0.3 5 $layer $emb_base 0.2 2e-5
# run_tacred_with_warmuphyper 0.3 short 0.3 5 $layer $emb_base 0.1 5e-5
python ~/env_config/sending_emails.py -c "$0 status: $?"
