set -x


model="bert_base"
entities_tsv="/home/lr/yukun/kg-bert/entities.slimed.tsv"

function gen_descrip_emb() {
    local data=$1
    local layer=$2
    local emb_base=$3
    local add_postion=$4

    python ./code/descrip_emb_util.py \
        --data_dir "$data" \
        --ernie_model $model \
        --entities_tsv $entities_tsv\
        --do_lower_case \
        --threshold 0.0 \
        --output_base "$emb_base" \
        --max_seq_length 10 \
        --add_postion $add_postion \
        --bert_layer $layer
}

function run_figer() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}"
    
    python code/run_figer_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base --do_train   \
        --do_lower_case   --data_dir "./data/FIGER"   --ernie_model $model   \
        --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5   \
        --num_train_epochs 3.0   --output_dir "tmp" \
        --loss_scale 128 --entities_tsv $entities_tsv --warmup_proportion 0.2
}

function run_tying() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
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
    local layer=$5
    local emb_base=$6
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}"
    
    python3 code/run_tacred_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/tacred"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   \
        --learning_rate 2e-5   --num_train_epochs 4.0  --output_dir "tmp"   \
        --loss_scale 128 --entities_tsv $entities_tsv
}

function run_tacred_with_warmuphyper() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
    local warmup=$7
    local lr=$8
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}warm${warmup}lr${lr}"
    
    python3 code/run_tacred_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/tacred"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   \
        --learning_rate $lr   --num_train_epochs 4.0  --output_dir "tmp"   \
        --loss_scale 128 --entities_tsv $entities_tsv  --warmup_proportion $warmup
}

function run_fewrel() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}"
    
    python3 code/run_fewrel_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/fewrel"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   --learning_rate 2e-5  \
        --num_train_epochs 12   --output_dir "tmp"      \
        --loss_scale 128 --entities_tsv $entities_tsv
}

function run_fewrel_with_warmuphyper() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=$4
    local layer=$5
    local emb_base=$6
    local warmup=$7
    local lr=$8
    local seed=$9
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}warm${warmup}lr${lr}seed${seed}"
    
    python3 code/run_fewrel_with_split_descrip.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/fewrel"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   --learning_rate $lr  \
        --num_train_epochs 10   --output_dir "tmp"      \
        --loss_scale 128 --entities_tsv $entities_tsv --warmup_proportion $warmup \
        --seed $seed
}

function run_fewrel_with_warmuphyper_random_entemb() {
    local tgt_thre=$1
    local sort=$2
    local thre=$3
    local max_parent=1
    local layer=$5
    local emb_base=$6
    local warmup=$7
    local lr=$8
    local seed=$9
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}sort${sort}warm${warmup}lr${lr}seed${seed}"
    
    python3 code/run_fewrel_with_split_descrip_random_ent_emb.py --note $note \
        --target_threshold $tgt_thre --sort $sort --threshold $thre \
        --max_parent $max_parent --emb_base $emb_base  --do_train   \
        --do_lower_case   --data_dir "./data/fewrel"   --ernie_model $model   \
        --max_seq_length 256   --train_batch_size 16   --learning_rate $lr  \
        --num_train_epochs 10   --output_dir "tmp"      \
        --loss_scale 128 --entities_tsv $entities_tsv --warmup_proportion $warmup \
        --seed $seed
}
