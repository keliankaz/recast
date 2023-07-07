# Flexible and Scalable Earthquake Forecasting
This repository includes the reference material for *Using deep-learning for flexible and scalable earthquake forecasting* by Kelian Dascher-Cousineau, Oleksandr Shchur, Emily Brodsky, and Stephan Günnemann. Neural Temporal Point Process (NTPP) models provide an alternative approach to earthquake forecasting. Here, present an implementation of an NTPP, the Recurrent Earthquake foreCAST (RECAST).

The code includes:

1. Model definitions for both ETAS and the NTPP implementation: RECAST in the `eq` Python library 
2. Scripts for model training in the featured experiments in `experiments/`
3. Tutorials explaining how the code works: `notebooks/1. Training the model.ipynb`, `notebooks/2. Forecasting.ipynb`, and  `notebooks/3. (Experimental) training with extra features.ipynb`
3. Jupyter Notebooks to reproduce figures 2-4 in `notebooks/generate_figures/`
4. The trained models in `trained_models/`

## Reproducing the results
To reproduce the experimental results from the paper, please see the file `REPRODUCE.md`.

## Installation
The code has been tested on Linux (Ubuntu 20.04) and MacOS.
1. Make sure you have the latest version of [`conda`](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Install the dependencies and create a new conda environment.
    ```bash
    cd earthquake-ntpp
    conda env create -f environment.yml
    conda activate eq
    ```
   If you don't have a GPU on your machine, remove the line 
    ```
      - cudatoolkit=11.3
    ```
    from the file `environment.yml` before executing the commands above.
3. Install the `eq` package.
    ```bash
    pip install -e .
    ```


## Examples
### Training the model
A simple example of training an NTPP model via maximum likelihood.
```python
import eq
import pytorch_lightning as pl

# Load the earthquake catalog from (White et al. 2020)
catalog = eq.catalogs.White()

# Create train, validation and test dataloaders
dl_train = catalog.train.get_dataloader()
dl_val = catalog.val.get_dataloader()
dl_test = catalog.test.get_dataloader()

# Initialize and train the model
ntpp_model = eq.models.RecurrentTPP()
trainer = pl.Trainer(gpus=1, max_epochs=200)
trainer.fit(ntpp_model, dl_train)

# Compute negative log-likelihood on the test set
nll_test = trainer.test(ntpp_model, dl_test)
```

### Generating a forecast
Use a trained model to sample possible continuations of the catalog
```python
import eq

# Load the catalog
catalog = eq.catalogs.White()
test_seq = catalog.test[0]

# Generate 1000 forecasts over interval [4000, 4007] given events from [0, 4000]
t_forecast = 4000
duration = 7
num_samples = 1000
# Past events that we condition on
past_seq = test_seq.get_subsequence(0, t_forecast)
# Observed events in the 7-day forecast window
observed_seq = test_seq.get_subsequence(t_forecast, t_forecast + duration)

# Load a trained model from the checkpoint
model = eq.models.RecurrentTPP.load_from_checkpoint("trained_models/White_RecurrentTPP.ckpt")
model.eval()

# Sample 1000 trajectories over the 7-day window
forecast = model.sample(batch_size=num_samples, duration=duration, past_seq=past_seq, return_sequences=True)

# Visualize the forecast
eq.visualization.visualize_trajectories(test_seq, forecast)
```

## Available models
- `eq.models.ETAS`: The classic Epidemic Type Aftershock Sequence (ETAS) model.
- `eq.models.RecurrentTPP`: Neural Temporal Point Process (NTPP) model proposed in the paper.

## Available earthquake catalogs
The code includes the following earthquake catalogs for Southern California that we used in our experiments.
| Catalog name                                                                                         | Catalog start | Catalog end | # events | Magnitude of completness |
| ---------------------------------------------------------------------------------------------------- | ------------- | ----------- | -------- | ------------------------ |
| [`eq.catalogs.White`](https://data.mendeley.com/datasets/7ywkdx7c62/1)                               | 2008-01       | 2021-01     | 134975   | 0.6                      |
| [`eq.catalogs.SCEDC`](https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/)                       | 1981-01       | 2020-01     | 125421   | 2.0                      |
| [`eq.catalogs.QTMSaltonSea`](https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_12dev.hypo)  | 2008-01       | 2018-01     | 44133    | 1.0                      |
| [`eq.catalogs.QTMSanJacinto`](https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_12dev.hypo) | 2008-01       | 2018-01     | 20790    | 1.0                      |

In addition, the following synthetic catalogs were used in the experiments
- `eq.catalogs.ETAS_SingleCatalog`: One long catalog produced by the ETAS model.
- `eq.catalogs.ETAS_MultiCatalog`: Multiple short catalogs produced by the ETAS model.

