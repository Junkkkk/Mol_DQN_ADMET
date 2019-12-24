# Mol-DQN
Optimization of Molecules via Deep Reinforcement Learning
https://arxiv.org/abs/1810.08678

##Model

1). logP
 - reward : penalized logP
 - ./models/logp_model/train.py

2). QED
 - reward : QED(Quantitative Estimate of Druglikeness)
 - ./models/qed_model/train.py

3). logP_constraint
 - reward : logP - w*(k-similarity)
 - ./models/logP_constraint/train.py

4). multiobjective
 - reward : w*similarity + (1-w)*QED
 - ./models/multi_logp_qed_model/train.py
 
 ##Result
 
