# Flexible and Scalable Earthquake Forecasting
This repository includes the reference material for *Using deep-learning for flexible and scalable earthquake forecasting* by Kelian Dascher-Cousineau, Oleksandr Shchur, Emily Brodsky, and Stephan Günnemann. Neural Temporal Point Process (NTPP) models provide an alternative approach to earthquake forecasting. Here, we present an implementation of an NTPP, the Recurrent Earthquake foreCAST (RECAST).

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
    cd recast
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
trainer = pl.Trainer(max_epochs=200)
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




UC Santa Cruz Noncommercial License

Acceptance

In order to get any license under these terms, you must agree to them as both strict obligations and conditions to all your licenses.

Copyright License

The licensor grants you a copyright license for the software to do everything you might do with the software that would otherwise infringe the licensor's copyright in it for any permitted purpose. However, you may only distribute the software according to Distribution License and make changes or new works based on the software according to Changes and New Works License.

Distribution License

The licensor grants you an additional copyright license to distribute copies of the software. Your license to distribute covers distributing the software with changes and new works permitted by Changes and New Works License.

Notices

You must ensure that anyone who gets a copy of any part of the software from you also gets a copy of these terms, as well as the following copyright notice:

This software is Copyright ©2023. The Regents of the University of California (“Regents”). All Rights Reserved.

Changes and New Works License

The licensor grants you an additional copyright license to make changes and new works based on the software for any permitted purpose.

Noncommercial Purposes

Any noncommercial purpose is a permitted purpose.

Commercial Purposes

Contact Innovation Transfer, UC Santa Cruz, innovation@ucsc.edu , https://officeofresearch.ucsc.edu/iatc/ , for any commercial purpose.

Personal Uses

Personal use for research, experiment, and testing for the benefit of public knowledge, personal study, private entertainment, hobby projects, amateur pursuits, or religious observance, without any anticipated commercial application, is use for a permitted purpose.

Noncommercial Organizations

Use by any charitable organization, educational institution, public research organization, public safety or health organization, environmental protection organization, or government institution is use for a permitted purpose regardless of the source of funding or obligations resulting from the funding.

Fair Use

You may have "fair use" rights for the software under the law. These terms do not limit them.

No Other Rights

These terms do not allow you to sublicense or transfer any of your licenses to anyone else, or prevent the licensor from granting licenses to anyone else.  These terms do not imply any other licenses.

Patent Defense

If you make any written claim that the software infringes or contributes to infringement of any patent, all your licenses for the software granted under these terms end immediately. If your company makes such a claim, all your licenses end immediately for work on behalf of your company.

Violations

The first time you are notified in writing that you have violated any of these terms, or done anything with the software not covered by your licenses, your licenses can nonetheless continue if you come into full compliance with these terms, and take practical steps to correct past violations, within 32 days of receiving notice.  Otherwise, all your licenses end immediately.

No Liability

As far as the law allows, the software comes as is, without any warranty or condition, and the licensor will not be liable to you for any damages arising out of these terms or the use or nature of the software, under any kind of legal claim.

Definitions

The "licensor" is Regents, and the "software" is the software the licensor makes available under these terms.

"You" refers to the individual or entity agreeing to these terms.

"Your company" is any legal entity, sole proprietorship, or other kind of organization that you work for, plus all organizations that have control over, are under the control of, or are under common control with that organization.  

"Control" means ownership of substantially all the assets of an entity, or the power to direct its management and policies by vote, contract, or otherwise.  Control can be direct or indirect.

"Your licenses" are all the licenses granted to you for the software under these terms.

"Use" means anything you do with the software requiring one of your licenses.

