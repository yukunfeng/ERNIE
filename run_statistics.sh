set -x

entities_tsv="/home/lr/yukun/kg-bert/entities.slimed.tsv"

python ./code/descrip_emb_util.py \
    --data_dir "./data/fewrel" \
    --ernie_model "none" \
    --entities_tsv $entities_tsv\
    --do_lower_case \
    --threshold 0.3 \
    --output_base "none" \
    --max_seq_length 10 \
    --bert_layer 2 \
    --statistics_mode "target" \
    --data_type "dev"
    # --statistics_mode "target"
