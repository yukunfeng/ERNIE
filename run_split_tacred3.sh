set -x

source ./run.sh
data="data/tacred"

layer=-2
emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base
run_tacred 0.3 short 0.3 5 $layer $emb_base
run_tacred 0.8 short 0.3 5 $layer $emb_base
run_tacred 100 short 0.3 5 $layer $emb_base
