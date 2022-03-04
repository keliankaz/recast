import math
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import sacred
import seml
import torch
import wandb

import eq
from eq.experiment_utils import split_minibatches, trim_train_and_test

ex = sacred.Experiment()
seml.setup_logger(ex)


@ex.config
def config_experiment():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


def compute_num_events(seq):
    return (seq.arrival_times > seq.t_nll_start).sum().item()


available_models = {
    "ETAS",
    "RecurrentTPP",
}

available_datasets = {
    "QTMSaltonSea",
    "QTMSanJacinto",
    "SCEDC",
    "White",
    "ETAS_MultiCatalog",
    "ETAS_SingleCatalog",
}

project_root_dir = Path(eq.__file__).parents[1]
# Filter annoying warnings by PytorchLightning
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*has `shuffle=True`, it is strongly*")


@ex.automain
def train_model(
    dataset_name="White",
    model_name="RecurrentTPP",
    context_size=32,
    num_components=32,
    dropout_proba=0.1,
    learning_rate=5e-2,
    batch_size=None,
    max_epochs=1500,
    patience=200,
    random_seed=0,
    use_gpu=True,
    use_double_precision=False,
    minibatch_training=False,
    train_fraction=None,
    log_dir=f"{project_root_dir / 'logs'}{os.sep}",
    model_save_path=None,
    wandb_entity=None,
    wandb_project=None,
):
    """Train a TPP model on the catalog and report NLL loss on train / val / test sets.

    Args:
        dataset_name: Name of the catalog. See available_datasets above for options.
        model_name: Name of the model, {"RecurrentTPP", "ETAS"}
        context_size: Size of the RNN hidden state (RecurrentTPP)
        num_components: Number of Weibull mixture components (RecurrentTPP)
        dropout_proba: Dropout probability (RecurrentTPP)
        learning_rate: Learning rate (both models)
        batch_size: Batch size. Only used for the ETAS_MultiCatalog (other catalogs
            contain only a single sequence).
        max_epochs: Maximum number of training epochs.
        patience: Early stopping if validation loss doesn't improve for this many epochs
        random_seed: Seed for all RNG (for reproducibility)
        use_gpu: If True, train on GPU, else on CPU
        use_double_precision: If True, train with double precision, else with single.
            This should be set to True for ETAS model: ETAS uses the arrival times, and
            loss of numerical precision might occur when training on long catalogs.
        minibatch_training: Break down long sequences into multiple shorter ones. This
            allows us to train ETAS on GPU on long catalogs.
        train_fraction: Fraction of the train / val sets used for training. Used in the
            experiment in Figure 3. If None, entire train / val sets are used.
        log_dir: Directory where the logs are saved.
        model_save_path: If provided, the trained model will be saved to this path.
        wandb_entity: Specify this to sync the run to wandb.ai.
        wandb_project: Specify this to sync the run to wandb.ai.
    """
    config = locals().copy()
    pl.seed_everything(random_seed)
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    if dataset_name not in available_datasets:
        raise ValueError(
            f"dataset_name must be one of {available_datasets} (got {dataset_name} instead)"
        )
    if model_name not in available_models:
        raise ValueError(
            f"dataset_name must be one of {available_models} (got {model_name} instead)"
        )

    # Load the dataset
    catalog = getattr(eq.catalogs, dataset_name)()

    # Convert precision to double, if necessary
    if use_double_precision:
        for cat in (catalog.train, catalog.val, catalog.test):
            for seq in cat:
                seq.double()
        precision = 64
    else:
        for cat in (catalog.train, catalog.val, catalog.test):
            for seq in cat:
                seq.float()
        precision = 32

    # Trim train and val sets, if necessary
    if train_fraction is not None:
        catalog = trim_train_and_test(
            catalog, train_frac=train_fraction, val_frac=train_fraction
        )

    num_events_train = sum(compute_num_events(seq) for seq in catalog.train)
    num_events_val = sum(compute_num_events(seq) for seq in catalog.val)

    tau_mean = torch.cat([seq.inter_times[:-1] for seq in catalog.train]).mean().item()
    mag_mean = torch.cat([seq.mag for seq in catalog.train]).mean().item()

    # Split into minibatches, if necessary
    if minibatch_training:
        print("Splitting into minibatches")
        catalog = split_minibatches(catalog)

    # Estimate the b parameter of the magnitude distribution
    if "richter_b" in catalog.metadata:
        # Use ground truth value, if available
        richter_b_mle = catalog.metadata["richter_b"]
    else:
        mag_roundoff_error = catalog.metadata.get("mag_roundoff_error", 0.0)
        richter_b_mle = math.log10(math.exp(1)) / (
            mag_mean - catalog.metadata["mag_completeness"] + 0.5 * mag_roundoff_error
        )

    if model_name == "ETAS":
        # ETAS training / evaluation is done with double precision
        dl_train = catalog.train.get_dataloader(batch_size=1, shuffle=False)
        dl_val = catalog.val.get_dataloader(batch_size=1)
        dl_test = catalog.test.get_dataloader(batch_size=1)
        model = eq.models.ETAS(
            learning_rate=learning_rate,
            richter_b=richter_b_mle,
            base_rate_init=1 / tau_mean,
            mag_completeness=catalog.metadata["mag_completeness"],
        )
        model.double()
        # Add random noise if random_seed != 0
        if random_seed != 0:
            for param in model.parameters():
                param.data += np.random.uniform(-0.3, 0.3)

        # ETAS model struggles with large batch sizes due to high memory consumption
        # we accumulate gradients to give it the same effective batch size as NTPP
        if batch_size is not None:
            accumulate_grad_batches = min(batch_size, len(catalog.train))
        else:
            accumulate_grad_batches = len(catalog.train)
    else:  # model_name == "RecurrentTPP":
        if batch_size is None:
            batch_size = 1
        dl_train = catalog.train.get_dataloader(batch_size=batch_size, shuffle=True)
        dl_val = catalog.val.get_dataloader(batch_size=len(catalog.val))
        dl_test = catalog.test.get_dataloader(batch_size=len(catalog.test))
        model = eq.models.RecurrentTPP(
            context_size=context_size,
            num_components=num_components,
            dropout_proba=dropout_proba,
            learning_rate=learning_rate,
            tau_mean=tau_mean,
            mag_mean=mag_mean,
            richter_b=richter_b_mle,
            mag_completeness=catalog.metadata["mag_completeness"],
        )
        accumulate_grad_batches = 1

    Path(log_dir).mkdir(exist_ok=True, parents=True)
    work_dir = tempfile.TemporaryDirectory(prefix=str(log_dir))
    print(f"Saving logs to {work_dir.name}")

    checkpoint = pl_callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=work_dir.name,
    )

    early_stopping = pl_callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=1e-3,
    )

    # Initialize W&B
    if wandb_entity is not None:
        wandb_run_name = f"{dataset_name}-{model_name}-{random_seed}"
        if train_fraction is not None:
            wandb_run_name += f"-trim_{train_fraction:.2f}"
        if minibatch_training:
            wandb_run_name += "-MINIBATCH"
        run = wandb.init(
            name=wandb_run_name,
            project=wandb_project,
            dir=work_dir.name,
            entity=wandb_entity,
            config=config,
        )
        logger = pl_loggers.WandbLogger(save_dir=work_dir.name)
    else:
        logger = False

    trainer = pl.Trainer(
        logger=logger,
        gpus=int(torch.cuda.is_available() and use_gpu),
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stopping],
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        enable_progress_bar=False,
    )

    trainer.fit(model, dl_train, val_dataloaders=dl_val)

    # Load the model with the best validation loss
    model = model.load_from_checkpoint(checkpoint.best_model_path)

    # Evaluate the model on CPU
    tester = pl.Trainer(
        gpus=int(torch.cuda.is_available() and use_gpu),
        logger=False,
        enable_checkpointing=False,
        precision=precision,
    )
    # We know that we're evaluating on CPU - ignore the warning
    warnings.filterwarnings("ignore", ".*GPU available but not used. Set the gpus *")
    final_nll_train = tester.test(model, dl_train)[0]["test_loss"]
    final_nll_val = tester.test(model, dl_val)[0]["test_loss"]
    final_nll_test = tester.test(model, dl_test)[0]["test_loss"]
    results = {
        "final_nll_train": final_nll_train,
        "final_nll_val": final_nll_val,
        "final_nll_test": final_nll_test,
    }

    if model_save_path is not None:
        print(f"Saving the best model weights to {model_save_path}")
        shutil.copyfile(checkpoint.best_model_path, model_save_path)

    # Sync results to W&B
    if wandb_entity is not None:
        results["wandb_url"] = run.get_url()[16:]
        wandb.log(
            {
                "final/nll_train": final_nll_train,
                "final/nll_val": final_nll_val,
                "final/nll_test": final_nll_test,
            }
        )
        # Copy the best model to the working directory & sync it to W&B
        best_model_path = Path(work_dir.name) / "best_model.ckpt"
        shutil.copyfile(checkpoint.best_model_path, best_model_path)
        wandb.save(str(best_model_path))
        wandb.finish()

    results["num_events_train"] = num_events_train
    results["num_events_val"] = num_events_val

    print(f"Deleting the logs directory {work_dir.name}")
    work_dir.cleanup()
    shutil.rmtree(work_dir.name, ignore_errors=True)
    return results
