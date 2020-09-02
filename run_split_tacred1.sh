set -x

source ./run.sh

layer=-2
# Run tacred
data="data/tacred"
emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base

run_tacred 100 short 100 1 $layer $emb_base

run_tacred 100 short 0 5 $layer $emb_base
run_tacred 100 short 0 1 $layer $emb_base

# run_tacred 100 short 0.3 5 $layer $emb_base
# run_tacred 100 short 0.3 1 $layer $emb_base

# run_tacred 0.3 short 0.3 5 $layer $emb_base
# run_tacred 0.3 short 0.3 1 $layer $emb_base
