set -x

# model="ernie_base"
model="bert_base"
# model="bert_standard"
# model="bert_descrip_same_time"
# model="bert_descrip_same_time_under_ernie"
# model="bert_descrip_same_time_contra"
# model="bert_descrip_as_input"

# data="data/FIGER_linked"
data="data/FIGER"

output="${model}_$(basename $data)_output"

python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/FIGER   --ernie_model $model   --max_seq_length 256   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir $output   --threshold 0.3  --loss_scale 128 --warmup_proportion 0.2
# evaluate
python3 code/eval_figer.py    --do_eval   --do_lower_case   --data_dir data/FIGER   --ernie_model $model   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir $output   --threshold 0.3  --loss_scale 128 --warmup_proportion 0.2


python ~/env_config/sending_emails.py -c "$0 $model finished. Start at 2020-07-13 20:51"
