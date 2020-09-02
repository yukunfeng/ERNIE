set -x

source ./run.sh

# Run typing
data="data/FIGER"

# layer=-2
# emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base
# run_figer 100 short 100 5 $layer $emb_base
# run_figer 100 short 0 5 $layer $emb_base
# run_figer 100 short 0 1 $layer $emb_base

# run_figer 100 short 0.3 5 $layer $emb_base
# run_figer 100 short 0.3 1 $layer $emb_base

# run_figer 0.3 short 0.3 5 $layer $emb_base
# run_figer 0.3 short 0.3 1 $layer $emb_base

# run_figer 0.6 short 0.3 5 $layer $emb_base
# run_figer 0.8 short 0.3 5 $layer $emb_base


layer=-1
emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base
# run_figer 0.6 long 0.3 5 $layer $emb_base
# run_figer 0.8 long 0.3 5 $layer $emb_base
run_figer 0.8 long 0.3 5 $layer $emb_base
# run_figer 100 long 0.3 5 $layer $emb_base

