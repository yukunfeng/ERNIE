set -x

source ./run.sh
function run_figer_with_lr() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
    local warmup=$7
    local lr=$8
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}warm${warmup}lr${lr}"
    
    python code/run_figer_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base --do_train   \
        --do_lower_case   --data_dir "./data/FIGER"   --ernie_model $model   \
        --max_seq_length 256 --train_batch_size 16 --learning_rate $lr   \
        --num_train_epochs 3.0   --output_dir "tmp" \
        --loss_scale 128 --entities_tsv $entities_tsv --warmup_proportion $warmup
}

# Run typing
data="data/FIGER"

# layer=-2
# emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base
# run_figer 100 short 100 5 $layer $emb_base
# run_figer 100 short 0 5 $layer $emb_base
# run_figer 100 short 0 1 $layer $emb_base

# run_figer 100 short 0.3 5 $layer $emb_base
# run_figer 100 short 0.3 1 $layer $emb_base

# run_figer 0.3 short 0.3 5 $layer $emb_base
# run_figer 0.3 short 0.3 1 $layer $emb_base

# run_figer 0.6 short 0.3 5 $layer $emb_base
# run_figer 0.8 short 0.3 5 $layer $emb_base


layer=-1
emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base
# run_figer 0.6 long 0.3 5 $layer $emb_base
# run_figer 0.8 long 0.3 5 $layer $emb_base
# run_figer 0.8 long 0.3 5 $layer $emb_base
# run_figer 0.3 short 0.3 5 $layer $emb_base
run_figer_with_lr 0.3 long 0.3 5 $layer $emb_base 0.2 4e-5

python ~/env_config/sending_emails.py -c "status: $? $0 figer, layer-2"

