python ./code/descrip_emb_util.py \
    --data_dir "data/fewrel" \
    --ernie_model "bert_base" \
    --entities_tsv "/home/lr/yukun/kg-bert/entities.slimed.tsv" \
    --do_lower_case \
    --threshold 0.0 \
    --output_base "fewrel_descrip_emb" \
    --max_seq_length 10 \
    --bert_layer -1
