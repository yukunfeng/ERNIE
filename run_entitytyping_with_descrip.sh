set -x

# model="ernie_base"
# model="bert_base_under_ernie"
# model="bert_wwm_base"
# model="bert-large-wwm"
# model="bert_standard"
# model="bert_descrip_same_time"
# model="bert_descrip_same_time_under_ernie"
# model="bert_descrip_same_time_contra"
# model="bert_descrip_as_input"
# model="roberta-base"
model="bert_base"

data="data/OpenEntity"

layer=-1
thre=100.0
tgt_thre=100.0
emb_base="openentity_descrip_emb_layer${layer}"
sort="short"

entities_tsv="/home/lr/yukun/kg-bert/entities.slimed.tsv"
output="${model}_$(basename $data)_descrip_output_layer${layer}"
rm -rf $output

# First generate descrip embs.
python ./code/descrip_emb_util.py \
    --data_dir "$data" \
    --ernie_model $model \
    --entities_tsv $entities_tsv\
    --do_lower_case \
    --threshold 0.0 \
    --output_base "$emb_base" \
    --max_seq_length 10 \
    --bert_layer $layer

max_parents=(5)


for max_parent in "${max_parents[@]}"
do
    note="P${max_parent}L${layer}tgtt${tgt_thre}thre${thre}"
    python code/run_openentity_with_split_descrip.py --note $note --target_threshold $tgt_thre --sort $sort --threshold $thre --max_parent $max_parent --emb_base $emb_base --do_train   --do_lower_case   --data_dir $data   --ernie_model $model   --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir $output --loss_scale 128 --entities_tsv $entities_tsv

    # python code/run_openentity_with_descrip.py --no_descrip --max_parent 1 --emb_base $emb_base --do_train   --do_lower_case   --data_dir $data   --ernie_model $model   --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir $output --loss_scale 128 --entities_tsv $entities_tsv
    # python code/run_openentity_with_split_descrip.py --threshold 0.0 --max_parent $max_parent --emb_base $emb_base --do_train   --do_lower_case   --data_dir $data   --ernie_model $model   --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir $output --loss_scale 128 --entities_tsv $entities_tsv
    # python code/run_openentity_with_descrip.py --threshold 0.0 --max_parent $max_parent --emb_base $emb_base --do_train   --do_lower_case   --data_dir $data   --ernie_model $model   --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir $output --loss_scale 128 --entities_tsv $entities_tsv

    # echo "above result, max_parent: $max_parent. layer:$layer with descrip"
done

# python ~/env_config/sending_emails.py -c "$0 succ: $? $model finished;"
