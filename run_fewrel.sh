set -x

# model="ernie_base"
# model="bert_base_under_ernie"
model="bert_base"
# model="bert_wwm_base"
# model="bert-large-wwm"
# model="bert_standard"
# model="bert_descrip_same_time"
# model="bert_descrip_same_time_under_ernie"
# model="bert_descrip_same_time_contra"
# model="bert_descrip_as_input"
# model="roberta-base"

data="data/fewrel"

# output="${model}_$(basename $data)_output"
output="${model}_$(basename $data)_output_debug"
rm -rf $output


python3 code/run_fewrel.py   --do_train   --do_lower_case   \
    --data_dir $data   --ernie_model $model   --max_seq_length 256   \
    --train_batch_size 16   --learning_rate 2e-5   \
    --num_train_epochs 10   --output_dir $output      --loss_scale 128 --threshold 0
# evaluate
python3 code/eval_fewrel.py   --do_eval   --do_lower_case   \
    --data_dir $data   --ernie_model $model   --max_seq_length 256   \
    --train_batch_size 32   --learning_rate 2e-5   \
    --num_train_epochs 10   --output_dir $output      --loss_scale 128 --threshold 0

python ~/env_config/sending_emails.py -c "$0 succ: $? $model finished;"
