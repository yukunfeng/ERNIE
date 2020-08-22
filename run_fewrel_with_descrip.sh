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

data="data/fewrel"

emb_base="fewrel_descrip_emb"
max_parent=1

entities_tsv="/home/lr/yukun/kg-bert/entities.slimed.tsv"
output="${model}_$(basename $data)_descrip_output_contra"
# output="${model}_$(basename $data)_output_debug"
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
    --bert_layer -1

python3 code/run_fewrel_with_descrip.py --max_parent $max_parent --emb_base $emb_base  --do_train   --do_lower_case   --data_dir $data   --ernie_model $model   --max_seq_length 256   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10   --output_dir $output      --loss_scale 128 --entities_tsv $entities_tsv
# evaluate
# python3 code/eval_fewrel.py   --do_eval   --do_lower_case   --data_dir $data   --ernie_model $model   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir $output      --loss_scale 128

# python ~/env_config/sending_emails.py -c "$0 succ: $? $model finished;"
