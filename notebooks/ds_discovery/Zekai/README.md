# Zekai Wang's Log

## Directory Walkthrough

```training_etas.ipynb, training_recurrent_attention.ipynb, training_recurrent.ipynb``` are notebooks training the three kinds of models with 
```
catalog = eq.catalogs.ANSS_MultiCatalog(num_sequences=10000,
    t_end_days=4*365,
    mag_completeness=4.5,
    minimum_mainshock_mag=6.0,)
```

```experiments``` contains many temporary files about the experiments. ```traind_etas, trained_recurrent_tpp``` the trained model in notebook ```training_etas.ipynb, training_recurrent.ipynb```. ```experiments/recurrent_attention_2023-10-19 13:11:44.713977``` contain the Tensorboard log of the experiment for ```training_recurrent_attention.ipynb```. 

Tensorboard Usage: in the terminal run ```tensorboard --logdir [directory_name]``` and then open ```http://localhost:6006/``` in a browser / the simple browser of VSCode. 


## Current TODOs

Right now I am working on letting ```recurrent_attention``` work. Possible next steps: 1. add normalization layer between attention and RNN; 2. Clip gradients; 3. get rid of the RNN completely (Attention only / Neural ODE / ...)