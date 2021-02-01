set -x

source ./run.sh

layer=-2
# Run fewrel
data="data/fewrel"
emb_base="$(basename $data)_random_entemb"
# gen_descrip_emb $data $layer $emb_base "0"

# with seed
array=(234)
for ((idx=0; idx<${#array[@]}; ++idx)); do
    run_fewrel_with_warmuphyper_random_entemb 0 long 0 1 $layer $emb_base 0.1 4e-5 "${array[idx]}"
done
