# functions adapted from:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html
# for compatability with equinox

# Standard libraries
import os
from typing import Any, Sequence, Optional, Iterator, Dict, Union
import json
import time
from tqdm.autonotebook import tqdm
import numpy as np
from copy import copy
from collections import defaultdict

# JAX/Equinox
import jax
from jax import random
import equinox as eqx
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

## Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns


def numpy_collate(batch):
    '''Collate function for compatability between torch dataloader and numpy array'''
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_data_loaders(
    *datasets: Sequence[data.Dataset],
    train: Union[bool, Sequence[bool]] = True,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Creates data loaders used in JAX for a set of datasets.

    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            collate_fn=numpy_collate,
            num_workers=num_workers,
            #  persistent_workers=is_train,
            generator=torch.Generator().manual_seed(seed),
        )
        loaders.append(loader)
    return loaders


class EqxTrainerModule:
    def __init__(
        self,
        model_class: eqx.Module,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 5,
        **kwargs,
    ):
        """
        A basic class summarizing most common training functionalities
        like logging, training loop, gradient evaluation, etc.

        Subclass must overwrite `create_functions` in order to train/eval models.

        Subclass can overwrite, if relevant, `vizualize_grad_norms`, `bind_model`, `on_training_start`,
          `on_training_epoch_end`, `on_validation_epoch_end`

        Atributes:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.key = random.key(seed)
        self.check_val_every_n_epoch = check_val_every_n_epoch
        # Set of hyperparameters to save
        self.config = {
            "model_class": model_class.__name__,
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": self.seed,
        }
        self.config.update(kwargs)
        # Create empty model. Note: no parameters yet
        self.key, subkey = random.split(self.key)
        self.model, self.state = eqx.nn.make_with_state(self.model_class)(
            key=subkey, **self.model_hparams
        )
        # Init trainer parts
        self.init_logger(logger_params)
        self.create_jitted_functions()

    def init_logger(self, logger_params: Optional[Dict]):
        """
        Initializes a logger and creates a logging directory.

        Args:
          logger_params: A dictionary containing the specification of the logger.
        """
        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get("log_dir", None)
        if not log_dir:
            base_log_dir = logger_params.get("base_log_dir", "checkpoints/")
            # Prepare logging
            log_dir = os.path.join(base_log_dir, self.config["model_class"])
            if "logger_name" in logger_params:
                log_dir = os.path.join(log_dir, logger_params["logger_name"])
            version = None
        else:
            version = ""
        # Create logger object
        logger_type = logger_params.get("logger_type", "TensorBoard").lower()
        if logger_type == "tensorboard":
            self.logger = TensorBoardLogger(save_dir=log_dir, version=version, name="")
        elif logger_type == "wandb":
            self.logger = WandbLogger(
                name=logger_params.get("project_name", None),
                save_dir=log_dir,
                version=version,
                config=self.config,
            )
        else:
            assert False, f'Unknown logger type "{logger_type}"'
        # Save hyperparameters
        log_dir = self.logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, "hparams.json")):
            os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
            with open(os.path.join(log_dir, "hparams.json"), "w") as f:
                json.dump(self.config, f, indent=4)
        self.log_dir = log_dir

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        elif optimizer_name.lower() == "nadam":
            opt_class = optax.nadam
        elif optimizer_name.lower() == "nadamw":
            opt_class = optax.nadamw
        elif optimizer_name.lower() == "lion":
            opt_class = optax.lion
        elif optimizer_name.lower() == "fromage":
            opt_class = optax.fromage
        elif optimizer_name.lower() == "lamb":
            opt_class = optax.lamb
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        # Initialize learning rate scheduler
        # A cosine decay scheduler is used, but others are also possible
        lr = hparams.pop("lr", 1e-3)
        max_decay = hparams.pop("max_decay", 0.01)
        scheduler = hparams.pop("scheduler", optax.warmup_cosine_decay_schedule)
        warmup = hparams.pop("warmup", 0)
        lr_schedule = scheduler(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=max_decay * lr,
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip_by_global_norm(hparams.pop("gradient_clip", 1.0))]
        if (
            opt_class == optax.sgd and "weight_decay" in hparams
        ):  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(hparams.pop("weight_decay", 0.0)))
        optimizer = optax.chain(*transf, opt_class(lr_schedule, **hparams))
        self.opt = optimizer
        self.opt_state = self.opt.init(eqx.filter(self.model, eqx.is_array))

    def create_jitted_functions(self):
        """
        Creates equinox jitted versions of the training and evaluation functions.
        If self.debug is True, jitting is not applied.
        """
        train_step, eval_step = self.create_functions()
        if self.debug:  # Skip jitting
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = eqx.filter_jit(train_step)
            self.eval_step = eqx.filter_jit(eval_step)

    def create_functions(self):
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function an updated model. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """

        def train_step(model: Any, opt_state: Any, batch: Any, key: Any, state: Any):
            metrics = {}
            return model, opt_state, metrics, state

        def eval_step(model: Any, batch: Any, key: Any, state: Any):
            metrics = {}
            return metrics

        raise NotImplementedError

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Optional[Iterator] = None,
        num_epochs: int = 500,
    ) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))

        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None
        for epoch_idx in self.tracker(
            range(1, num_epochs + 1), desc="Epochs", position=0
        ):
            train_metrics = self.train_epoch(train_loader)
            self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)
        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)
        # Close logger
        self.logger.finalize("success")
        return best_eval_metrics

    def train_epoch(self, train_loader: Iterator) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        self.key, *subkeys = random.split(self.key, num_train_steps + 1)
        for batch, k in self.tracker(
            zip(train_loader, subkeys),
            desc="Training",
            leave=False,
            position=1,
            disable=True,
        ):
            self.model, self.opt_state, step_metrics, self.state = self.train_step(
                self.model, self.opt_state, batch, k, self.state
            )
            for key in step_metrics:
                metrics["train/" + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def bind_model(self):
        """
        Returns a batched model with parameters bound to it. Enables an easier inference
        access.

        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        raise NotImplementedError

    def eval_model(
        self, data_loader: Iterator, log_prefix: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        self.key, *subkeys = random.split(self.key, len(data_loader) + 1)
        for batch, k in zip(data_loader, subkeys):
            step_metrics = self.eval_step(self.model, batch, k, self.state)
            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        return metrics

    def is_new_model_better(
        self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]
    ) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model.
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
          True if the new model is better than the old one, and False otherwise.
        """
        if old_metrics is None:
            return True
        for key, is_larger in [
            ("val/val_metric", False),
            ("val/acc", True),
            ("val/loss", False),
        ]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f"No known metrics to log on: {new_metrics}"

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Wraps an iterator in a progress bar tracker (tqdm) if the progress bar
        is enabled.

        Args:
          iterator: Iterator to wrap in tqdm.
          kwargs: Additional arguments to tqdm.

        Returns:
          Wrapped iterator if progress bar is enabled, otherwise same iterator
          as input.
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(
        self, epoch_idx: int, eval_metrics: Dict[str, Any], val_loader: Iterator
    ):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader: Data loader of the validation set, to support additional
            evaluation.
        """
        pass

    def save_model(self, step=None):
        """
        Saves current model to logging directory.
        """
        eqx.tree_serialise_leaves(
            os.path.join(self.log_dir, "best_mod.eqx"), (self.model, self.state)
        )

    def load_model(self):
        """
        Loads model from the logging directory.
        """
        self.model, self.state = eqx.tree_deserialise_leaves(
            os.path.join(self.log_dir, "best_mod.eqx"), (self.model, self.state)
        )

    def vizualize_grad_norms(self, batch):
        """Plots histogram of gradient norms of model leaves. Not implemented,
        but only `loss_function` needs to be filled in when overwriting. 
        
        Must return in format `loss, ({auxillary vars})"""

        def loss_function(model, x, y, key, state, train=True):
            model_out, state = jax.vmap(
                model,
                axis_name="batch",
                in_axes=(0, None, None, None),
                out_axes=(0, None),
            )(x, key, state, train)

            raise NotImplementedError

        def get_grad_and_path(model, batch, key, state):
            x, y = batch
            _, grads = eqx.filter_value_and_grad(loss_function, has_aux=True)(
                model, x, y, key, state
            )
            dt = jax.tree_util.tree_leaves_with_path(grads)
            grads, names = [], []
            for (
                path,
                grad,
            ) in (
                dt
            ):  # filter basic info we don't care about (bias terms, normalization)
                if (
                    (len(grad.shape) > 1)
                    and (path[-1].name != "bias")
                    and (not path[-2].name.endswith("norm"))
                ):
                    grads.append(grad.reshape(-1))
                    names.append(path[-2].name + "_" + path[-1].name)
            return grads, names

        def viz_grads(grads, names, color="C0"):
            columns = len(grads)
            fig, ax = plt.subplots(1, columns, figsize=(columns * 3.5, 2.5))
            for g_idx, g in enumerate(grads):
                key = f"{names[g_idx]}"
                key_ax = ax[g_idx % columns]
                sns.histplot(data=g, bins=30, ax=key_ax, color=color, kde=True)
                key_ax.set_title(str(key))
                key_ax.set_xlabel("Grad magnitude")
            fig.suptitle("Gradient magnitude distributions", fontsize=14, y=1.05)
            fig.subplots_adjust(wspace=0.45)
            plt.show()
            plt.close()

        grads, names = get_grad_and_path(self.model, batch, self.key, self.state)
        viz_grads(grads, names)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str,
    ) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory.

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          exmp_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file"
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        hparams.pop("model_class")
        hparams.update(hparams.pop("model_hparams"))
        if not hparams["logger_params"]:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        trainer = cls(**hparams)
        trainer.load_model()
        return trainer
