# TODO
- Make sure figures / tables are aligned with the paper
- Make sure all figures can be reproduced

# Reproducing the results from the paper
Since a large number of experiments have to be run, we provide the following options to reproduce the results:
1. Use [`seml`](https://github.com/TUM-DAML/seml) to run the experiments via [`Slurm`](https://slurm.schedmd.com/overview.html), as we did.
Make sure you specify your own Slurm partition and Weights&Biases entity/project name if you choose this option.
2. Run individual experiments locally. This approach doesn't scale but can be used to verify individual results.

The intermediate results can also be found in `results/`. You can use them to directly generate the figures. For this, just execute the respective Jupyter notebooks in `notebooks/generate_figures/`.

## Training on full catalogs (Figure 2 + Table S1)
### With `seml`
1. Submit the jobs
    ```
    seml eq_mle add -ncc experiments/train_model/config_etas.yaml
    seml eq_mle add -ncc experiments/train_model/config_ntpp.yaml
    seml eq_mle start
    ```
2. Collect the results by running `notebooks/collect_results/full_catalog_training.ipynb`.
3. Generate figures with `notebooks/generate_figures/synthetic.ipynb`

### Individual experiments

- Training the neural TPP model
    ```
    python experiments/train_model/train_model.py with dataset_name=White model_name=RecurrentTPP
    ```

- Training the ETAS model
    ```
    python experiments/train_model/train_model.py with dataset_name=White model_name=ETAS use_double_precision=True minibatch_training=True
    ```

See the description in `experiments/train_model/train_model.py` for more details and command line arguments.


## Training on trimmed catalogs (Figure 3 + Figure S2)
### With `seml`
1. Submit the jobs
    ```
    seml eq_trimmed add -ncc experiments/train_model/config_trimmed_etas.yaml
    seml eq_trimmed add -ncc experiments/train_model/config_trimmed_ntpp.yaml
    seml eq_trimmed start
    ```
2. Collect the results by running `notebooks/collect_results/ETAS_SingleCatalog_trimmed.ipynb` and `notebooks/collect_results/White_trimmed.ipynb`.
3. Generate figures with `notebooks/generate_figures/trimmed.ipynb`.

### Individual experiments
- Training the neural TPP model
    ```
    python experiments/train_model/train_model.py with dataset_name=White model_name=RecurrentTPP train_fraction=0.1
    ```

- Training the ETAS model
    ```
    python experiments/train_model/train_model.py with dataset_name=White model_name=ETAS use_double_precision=True minibatch_training=True train_fraction=0.1
    ```

See the description in `experiments/train_model/train_model.py` for more details and command line arguments.

## Forecasting (Figure 4)
### With `seml`
1. Submit the jobs
    ```
    seml eq_forecast add -ncc experiments/forecast/config.yaml
    seml eq_forecast start
    ```
2. Generate the figures with `notebooks/generate_figures/forecasting.ipynb`.

### Individual experiments
```
python experiments/forecast/make_forecast.py with time_index=10 dataset_name=SCEDC model_name=ETAS forecast_duration=14 num_samples=10000
```
Results are saved to `results/samples_{dataset_name}_t{forecast_duration}_{num_samples}/{dataset_name}_{model_name}_{time_index}.npy`

See the docstring in `experiments/forecast/make_forecast.py` for more details and command line arguments.