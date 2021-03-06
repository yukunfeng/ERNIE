set -x

source ./run.sh

layer=-1
# Run fewrel
data="data/fewrel"
emb_base="$(basename $data)_descrip_emb_layer${layer}"
gen_descrip_emb $data $layer $emb_base
# run_fewrel 100 short 100 1 $layer $emb_base

# run_fewrel 100 short 0 5 $layer $emb_base
# run_fewrel 100 short 0 1 $layer $emb_base

run_fewrel 0 long 0 5 $layer $emb_base
# run_fewrel 0 short 0 1 $layer $emb_base
