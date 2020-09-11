set -x

source ./run.sh

layer=-2
# Run fewrel
data="data/fewrel"
emb_base="$(basename $data)_descrip_emb_layer${layer}"
# gen_descrip_emb $data $layer $emb_base 1

# run_fewrel_with_warmuphyper 0 short 100 5 $layer $emb_base 0.1 4e-5
run_fewrel_with_warmuphyper 0 long 100 5 $layer $emb_base 0.1 4e-5
# run_fewrel_with_warmuphyper 100 short 100 5 $layer $emb_base 0.1 4e-5
# run_fewrel 0 short 0 5 $layer $emb_base
# run_fewrel 0.2 short 0.2 5 $layer $emb_base
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.02 2e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.1 4e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.1 4e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.1 5e-5
# run_fewrel_with_warmuphyper 100 short 0 5 $layer $emb_base 0.1 4e-5
# run_fewrel_with_warmuphyper 100 short 100 5 $layer $emb_base 0.1 4e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.1 4e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.1 3e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.1 1e-5
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.05
# run_fewrel_with_warmuphyper 0 short 0 5 $layer $emb_base 0.15


python ~/env_config/sending_emails.py -c "$0 status: $?"
