set -x

# use fp16
# python code/run_typing.py    --do_train   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --fp16 --loss_scale 128
# not use fp16
# python code/run_typing.py    --do_train   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --loss_scale 128

# model="ernie_base"
# model="bert_base_under_ernie"
model="bert_base"
# model="bert_wwm_base"
# model="bert_standard"
# model="bert_descrip_same_time"
# model="bert_descrip_same_time_contra"
# model="bert_descrip_as_input"

data="data/OpenEntity"

output="${model}_$(basename $data)_output"
rm -r $output

# max len 256
python code/run_typing.py    --do_train   --do_lower_case   --data_dir $data  \
    --ernie_model $model   --max_seq_length 256 --train_batch_size 16 \
    --learning_rate 2e-5   --num_train_epochs 10.0   \
    --output_dir $output --threshold 0.3 --loss_scale 128

# evaluate
# max len 256 
python code/eval_typing.py   --do_eval   --do_lower_case   --data_dir $data   \
    --ernie_model $model   --max_seq_length 256   --train_batch_size 32   \
    --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir $output \
    --threshold 0.3 --fp16 --loss_scale 128

# max len 128
# python code/eval_typing.py   --do_eval   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 128   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir $output --threshold 0.3 --fp16 --loss_scale 128
python ~/env_config/sending_emails.py -c "$0 succ: $? $model finished;"
