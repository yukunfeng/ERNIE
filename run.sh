set -x


model="bert_base"
entities_tsv="/home/lr/yukun/kg-bert/entities.slimed.tsv"

function gen_descrip_emb() {
    local data="$data"
    local layer=$2
    local emb_base=$3

    python ./code/descrip_emb_util.py \
        --data_dir "$data" \
        --ernie_model $model \
        --entities_tsv $entities_tsv\
        --do_lower_case \
        --threshold 0.0 \
        --output_base "$emb_base" \
        --max_seq_length 10 \
        --bert_layer $layer
}

function run_tying() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}"
    
    python code/run_openentity_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base --do_train   \
        --do_lower_case   --data_dir "./data/OpenEntity"   --ernie_model $model   \
        --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5   \
        --num_train_epochs 10.0   --output_dir "tmp" \
        --loss_scale 128 --entities_tsv $entities_tsv
}

function run_tacred() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}"
    
    python3 code/run_tacred_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/tacred"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   \
        --learning_rate 2e-5   --num_train_epochs 4.0  --output_dir "tmp"   \
        --loss_scale 128 --entities_tsv $entities_tsv
}


function run_fewrel() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}"
    
    python3 code/run_fewrel_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/fewrel"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   --learning_rate 2e-5  \
        --num_train_epochs 10   --output_dir "tmp"      \
        --loss_scale 128 --entities_tsv $entities_tsv
}

# Run typing
data="data/OpenEntity"
emb_base="$(basename $data)_descrip_emb_layer${layer}"
gen_descrip_emb $data -2 $emb_base
run_tying 100 short 100 100 5 
