## File Structure

**runs** - stores evaluation results, gifs, and graphs from runs; runs/ablations/_plots stores the graphs while every other folder in runs/albations stores the results of an experiment

**extract_metrics.py** - prints eval mean reward for all 3 seeds of a specified experiment; run by using *python extract_metrics.py [experiment name]*

**plot_seeds.py** - saves a plot of the mean learning curve the and mean eval 
reward for all experiments

**run-ablations.ps1** - runs all ablation combinations conseuctively

**train.py** - trains the models and saves evaluation results and gifs into *runs* folder

