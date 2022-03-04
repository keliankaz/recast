import math
from pathlib import Path

import numpy as np
import seml
import torch
from sacred import Experiment

import eq

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config_experiment():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


project_root_path = Path(eq.__file__).parents[1]


precision = {
    "SCEDC": 0.1,
    "QTMSanJacinto": 0.01,
    "QTMSaltonSea": 0.01,
    "White": 0.01,
}


def evaluate_forecast(model, seq, t_start, duration, num_samples=10):
    s_cond = seq.get_subsequence(seq.t_start, t_start)
    s_obs = seq.get_subsequence(t_start, t_start + duration)
    with torch.no_grad():
        samples = model.sample(
            num_samples, duration, past_seq=s_cond, return_sequences=True
        )
    n_obs = len(s_obs)
    n_sampled = np.array([len(s) for s in samples])
    quantile = np.mean(n_sampled < n_obs)
    likelihood = np.mean(n_sampled == n_obs)
    return n_obs, n_sampled, quantile, likelihood


@ex.automain
def run_experiment(
    time_index,
    dataset_name="SCEDC",
    model_name="ETAS",
    init_time=0.0,
    forecast_length=14,
    num_samples=50_000,
):
    """Generate forecast for the specified time interval in the test set of the catalog.

    Trained model is loaded from trained_models/{dataset_name}_{model_name}.ckpt

    Results are saved to `results/samples_{dataset_name}_t{forecast_duration}_{num_samples}/{dataset_name}_{model_name}_{time_index}.npy`

    Args:
        time_index: Number of interval in the test set on which to perform the forecast.
            For example, if `forecast_duration=14`, then `time_index=0` performs
            forecast on the first 14-day interval of the test set, `time_index=1`
            performs forecast on the second 14-day interval, etc. Possible values are
            from `0` to `floor((test_seq.t_end - test_seq.t_nll_start) / forecast_duration)`
        dataset_name: Name of the dataset.
        model_name: Name of the model.
        init_time: Samples are conditioned on the past events from the interval
            [init_time, test_seq.t_nll_start] of the test sequence.
        forecast_length: Duration of the forecast window (days).
        num_samples: Number of trajectories to simulate.
    """
    catalog = getattr(eq.catalogs, dataset_name)()
    ckpt_path = (
        project_root_path / "trained_models" / f"{dataset_name}_{model_name}.ckpt"
    )
    model = getattr(eq.models, model_name).load_from_checkpoint(ckpt_path)
    # Estimate the b parameter of the magnitude distribution
    mag_mean = torch.cat([seq.mag for seq in catalog.train]).mean().item()
    richter_b_mle = math.log10(math.exp(1)) / (
        mag_mean - catalog.metadata["mag_completeness"] + 0.5 * precision[dataset_name]
    )
    if model_name == "ETAS":
        model.b.data.data.fill_(richter_b_mle)
    elif model_name == "RecurrentTPP":
        model.richter_b.data.fill_(richter_b_mle)
    model.double()
    model.eval()

    seq = catalog.test[0].get_subsequence(init_time, catalog.test[0].t_end).double()
    test_start = seq.t_nll_start
    test_end = seq.t_end - forecast_length
    start_times = np.arange(test_start, test_end, step=forecast_length)

    t_start = start_times[time_index]

    print(f"Starting sampling for {t_start}")
    n_obs, n_sampled, quantile, likelihood = evaluate_forecast(
        model, seq, t_start, forecast_length, num_samples
    )

    result = {
        "time_index": time_index,
        "t_start": t_start,
        "forecast_length": forecast_length,
        "n_obs": n_obs,
        "n_sampled": list(n_sampled),
        "quantile": quantile,
        "likelihood": likelihood,
        "model_name": model_name,
        "dataset_name": dataset_name,
    }
    save_dir = (
        project_root_path
        / "results"
        / f"samples_{dataset_name}_t{forecast_length}_n{num_samples}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving results to {save_dir}")
    np.save(save_dir / f"{dataset_name}_{model_name}_{time_index:02d}.npy", result)
    del result["n_sampled"]
    return result
