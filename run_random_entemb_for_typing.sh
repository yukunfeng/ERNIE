set -x

source ./run.sh

# Run typing
data="data/OpenEntity"

layer=-2
emb_base="$(basename $data)_descrip_emb_layer${layer}_random_entemb"
gen_descrip_emb $data $layer $emb_base 0
run_tying_random_entemb 0.3 long 0.3 1 $layer $emb_base
