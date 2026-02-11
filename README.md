# File Structure

**runs** - stores evaluation results, gifs, and graphs from runs; runs/ablations/_plots stores the graphs, while every other folder in runs/ablations stores the results of an experiment; each experiment has 3 seeds, each storing gifs, evaluation metrics, and a tensorboard file for  graphs

**extract_metrics.py** - prints eval mean reward for all 3 seeds of a specified experiment; run by using *python extract_metrics.py [experiment name]*

**plot_seeds.py** - saves a plot of the mean learning curve and the mean eval 
reward for all experiments

**run-ablations.ps1** - runs all ablation combinations conseuctively

**train.py** - trains the models and saves evaluation results and gifs into the *runs* folder

# Writeup

## Best Run
![best (2)](https://github.com/user-attachments/assets/0eee51e4-191c-4efc-a93d-0a4022d6fed3)

## Compute Resources

* Intel(R) Core(™) Ultra 9 185H
* Base Speed: 2.30 GHz
* Sockets: 1
* Cores: 16
* Logical processors:22
* Virtualization: Enabled
* L1 cache: 1.6MB
* L2 cache: 18.0 MB
* L3 cache: 24.0 MB

## Techniques Used

Random Network Distillation (RND)
	In the baseline, exploration can die down early, especially since the code has a reward for staying alive (even if it’s removed after 250,000 steps). In addition, efficient ways of moving are developed relatively late in the baseline model. RND encourages exploration by generating intrinsic rewards based on state novelty (Raffin et al., 2022). This extra incentive for exploration speeds up the process of discovering efficient movement and decreases the chance of getting stuck in a bad local optimum. 

Generalized State-Dependent Exploration (gSDE) 
	Based on gifs of its intermediate steps, the baseline exploration is very twitchy and struggles to sustain partial gaits long enough to learn them. gSDE changes exploration so that noise is more consistent over several steps, as opposed to randomizing every step (Burda et al., 2019). Implementing gSDE made the model better and more consistent at exploration.

## Ablation Studies (2,000,000 steps each seed)






### Graphs

<img width="320" height="240" alt="bar_best_mean" src="https://github.com/user-attachments/assets/b473c69f-f584-4793-a173-c5876fef1835" />
<img width="320" height="240" alt="learning_curves" src="https://github.com/user-attachments/assets/e6e2b6d2-7691-4543-8982-ca65eaedd690" />

### Eval Mean Reward
| Config       | Seed 0 | Seed 1 | Seed 2 | Mean ± Std   |
| ------------ | -----: | -----: | -----: | ------------ |
| Base         |   55.2 |  141.3 |  242.6 | 146.4 ± 76.6 |
| + gSDE       |  211.2 |  234.7 |  202.7 | 183.8 ± 57.4 |
| + RND        |  134.4 |  303.5 |  156.6 | 198.2 ± 75.0 |
| + gSDE + RND |  268.9 |  261.9 |  267.8 | 266.2 ± 3.1  |



### Issues Encountered

The base PPO does not stay alive long enough to discover walking, often just falling onto the ground almost immediately. I tried to fix this by adding rewards for staying alive and maintaining a high vertical height. This, however, caused the walker to discover a local optimum by staying still and simply staying alive. I then removed the reward for height and had the staying alive reward removed after 250,000 steps. After these adjustments, the model could now walk reasonably competently.

I identified two areas of potential improvement from this baseline: stability (the baseline model sometimes made catastrophic policy shifts) and exploration. After searching the literature, I implemented early stopping (KLE-Stop), RND, and gSDE. KLE-Stop aimed to improve stability, RND aimed to improve exploration, and gSDE aimed to improve both. After testing all three improvements, however, I found that KLE-Stop and its combinations with other improvements performed substantially worse than the baseline model, while the gSDE + RND model had adequate stability. Because of this, I removed KLE-Stop, ending up with the final model.

### Conclusion

Minor reward shaping by temporarily encouraging survival was valuable in accelerating learning in the early steps, although it was important to not overshape. gSDE and RND both substantially improved performance over the baseline, both individually and especially combined. The final model was also notably much more consistent between seeds compared to the earlier models.

In the future, the model could be modified to move faster. The learning curve for the final model plateaued fairly early on, suggesting that running more steps would not notably increase performance. A potential solution is to encourage exploration even more, so the model finds a more efficient method of movement.

### References 
Burda, Y., Edwards, H., Storkey, A., and Klimov, O. 2019. Exploration by Random Network Distillation. International Conference on Learning Representations (ICLR). 
Raffin, A., Kober, J., and Stulp, F. 2022. Smooth Exploration for Robotic Reinforcement Learning. In Proceedings of the 5th Conference on Robot Learning (CoRL), Proceedings of Machine Learning Research (PMLR), 164:1634–1644.
