
# buds
bash exp/scripts/trainer.sh --gpu 0 --al buds --ll ft --sc kitchenEM --start_seed 0 --num_exps 1 --j 1
bash exp/scripts/trainer.sh --gpu 0 --al buds --ll er10 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1
bash exp/scripts/trainer.sh --gpu 0 --al buds --ll append4 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1

# ptgm 
bash exp/scripts/trainer.sh --gpu 0 --al ptgm --ll ft --sc kitchenEM --start_seed 0 --num_exps 1 --j 1
bash exp/scripts/trainer.sh --gpu 0 --al ptgm --ll er10 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1
bash exp/scripts/trainer.sh --gpu 0 --al ptgm --ll append4 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1

# ptgm + silc
bash exp/scripts/trainer.sh --gpu 0 --al silc --ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1

# variants
bash exp/scripts/trainer.sh --gpu 0 --al silc --ll conf99/ptgm_er10/s20g20b4/ptgm/s20g20b4 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1
bash exp/scripts/trainer.sh --gpu 0 --al silc --ll conf99/ptgm_ft/s20g20b4/ptgm/s20g20b4 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1
bash exp/scripts/trainer.sh --gpu 0 --al silc --ll conf99/ptgm_ft/s20g20b4/ptgm/s20g20b4 --sc kitchenEM --start_seed 0 --num_exps 1 --j 1

