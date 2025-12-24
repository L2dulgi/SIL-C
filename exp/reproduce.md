

# Main Table 1 (Skill-Policy Compatibility : Backward and Forward)
```bash
# Kitchen Emergent SIL (SIL-C)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM

# Kitchen Explicit SIL (SIL-C)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEX

# Appendix Table 12 (MT; experience replay with full replay)
python exp/trainer.py -al lazysi -ll conf99/ptgm_er100/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
```

# Main Table 2 (Sample Efficiency : Downstream Few-shot Imitation Learning)
```bash 
# 5 shot 
python exp/trainer.py -al lazysi -ll few5/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# 3 shot 
python exp/trainer.py -al lazysi -ll few3/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# 1 shot 
python exp/trainer.py -al lazysi -ll few1/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# 1 shot 50% Ratio
python exp/trainer.py -al lazysi -ll few1frac2/ptgm_append4/s20g20b4/instance/g20b1 -sc kitchenEM
# 1 shot 20% Ratio
python exp/trainer.py -al lazysi -ll few1frac5/ptgm_append4/s20g20b4/instance/g20b1 -sc kitchenEM
```

# Figure 4 (Modularity: Under Varying Design Choices for Hierarchical Argchitecture)
```bash
# Kitchen Emergent SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append4/g20b4/ptgm/s20g20b4 -sc kitchenEM

# Kitchen Explicit SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append16/g20b4/ptgm/s20g20b4 -sc kitchenEX

# Kitchen Emergent SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM

# Kitchen Explicit SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append16/s20g20b4/ptgm/s20g20b4 -sc kitchenEX



# Kitchen Emergent SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append4/g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEM

# Kitchen Explicit SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append16/g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEX

# Kitchen Emergent SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEM

# Kitchen Explicit SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append16/s20g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEX
```


# Table 3 (Robustness)
```bash
# noise X 1 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.01 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
# noise X 2 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.02 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
# noise X 3 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.03 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
# noise X 5 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.05 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
```

# Figure 6 Skill and Subtask Space Resolution 
```bash
## skill 10
# subtask 10
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s10g20b4/ptgm/s10g20b4 -sc kitchenEM
# subtask 20
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s10g20b4/ptgm/s20g20b4 -sc kitchenEM
# subtask 40
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s10g20b4/ptgm/s40g20b4 -sc kitchenEM

## skill 20
# Subtask 10
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s10g20b4 -sc kitchenEM
# Subtask 20
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# Subtask 40
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s40g20b4 -sc kitchenEM

## skill 40
# Subtask 10
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s40g20b4/ptgm/s10g20b4 -sc kitchenEM
# Subtask 20
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s40g20b4/ptgm/s20g20b4 -sc kitchenEM
# Subtask 40
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s40g20b4/ptgm/s40g20b4 -sc kitchenEM
```


# Appendix C.2 datastream sequence permutation
```bash
# permutation 1(original)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  kitchenEM
# permutation 2 
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  objective_p1
# permutation 3
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  objective_p2
# permutation 4
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  objective_p3
```