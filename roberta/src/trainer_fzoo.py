"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

from opacus.accountants.utils import get_noise_multiplier


_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def dpzero_clip(loss_diff, C=1.):
    tmp = torch.min(torch.ones_like(loss_diff), torch.div(C * torch.ones_like(loss_diff), torch.abs(loss_diff)))
    return torch.mul(tmp, loss_diff)

def fzoo_clip(d, C):
    norms = d.norm(dim=0, keepdim=True)
    factor = torch.clamp(C / (norms + 1e-12), max=1.0)
    return d * factor

class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad


    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()


    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.randint(0, 2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype) * 2 - 1
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps
        return model

    
    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector


    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None

        # set up dp parameters:
        if self.args.dpzero:
            sample_rate= total_train_batch_size / (model.num_k *  model.num_labels)
            multiplier = get_noise_multiplier(target_epsilon=self.args.dp_epsilon,
                                              target_delta=self.args.dp_delta,
                                              steps=t_total,
                                              sample_rate=sample_rate,
                                              )
            self.dpzero_gaussian_std = 2 * multiplier * self.args.dpzero_clip_threshold / total_train_batch_size
            self.noise_gen = torch.Generator()  # separate from the global RNG (never reseeded by torch.manual_seed)
            self.noise_gen.seed()

        # Optional: evaluate the N perturbed models in ONE batched forward pass (see src/zo_parallel.py)
        self.zp = None
        if getattr(self.args, "fzoo_parallel", False):
            from src.zo_parallel import ZOParallel
            assert not self.args.layer_wise_optim, 'parallel FZOO needs a fixed set of optimized parameters'
            _optim = [(n, p) for n, p in model.named_parameters() if self.should_optim(n, p)]
            self.zp = ZOParallel(model, _optim, n=self.args.fzoo_n, eps=self.args.zero_order_eps)
            logger.info(f"Parallel FZOO: N={self.args.fzoo_n}, {len(self.zp.cover)} parameter tensors perturbed in the batched forward; "
                        f"not perturbed (must not affect the loss): {self.zp.uncovered}")

        # ===== MOMENTUM STATE (Nesterov) ================================================================
        # beta = 0 -> momentum off: no buffer is allocated and the update below is exactly the original one.
        # beta comes from args.fzoo_momentum if run_fzoo.py defines that flag, otherwise from $FZOO_MOMENTUM.
        self.fzoo_beta = getattr(self.args, "fzoo_momentum", None)
        if self.fzoo_beta is None:
            self.fzoo_beta = float(os.environ.get("FZOO_MOMENTUM", "0"))
        assert 0.0 <= self.fzoo_beta < 1.0, "fzoo momentum must be in [0, 1)"
        # previous steps' information lives ONLY here: one fp32 velocity tensor per optimized parameter
        # (Parameter -> tensor, created lazily on the first step, fresh for every run; ~1.4 GB for roberta-large)
        self.fzoo_mom = {}
        if self.fzoo_beta > 0:
            logger.info(f"FZOO Nesterov momentum: beta={self.fzoo_beta}")
        # =================================================================================================

        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                self.named_parameters_to_optim = []
                for name, param in model.named_parameters():
                    if self.should_optim(name, param):
                        self.named_parameters_to_optim.append((name, param))

                assert self.args.dpzero
                assert self.args.efficient_zero_order
                assert not self.args.zero_order_use_trainer_optim
                assert self.args.gradient_accumulation_steps == 1
                assert self.args.zero_order_sample_scheduler is None
                assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

                # get N directions 
                N = self.args.fzoo_n
                random_seeds = []
                loss1s = []

                if self.zp is not None:
                    # batched: all N perturbed models in one forward pass (batch replicated N times)
                    with torch.no_grad():
                        random_seeds = [np.random.randint(1000000000) for _ in range(N)]
                        loss0 = self.zo_forward(model, inputs)                      # (B,) unperturbed, per example
                        self.zp.set_directions(random_seeds)                        # N int8 +-1 directions
                        with self.zp.perturbed():
                            loss_n = self.zo_forward(model, self.zp.replicate(inputs))   # (N*B,)
                    d = loss_n.view(N, -1) - loss0.unsqueeze(0)                     # (N, B)
                else:
                    with torch.no_grad():
                        loss0 = self.zo_forward(model, inputs) # initial loss value (will compare to others)
                        for i in range(N):
                            seed = np.random.randint(1000000000) # generate random seed
                            random_seeds.append(seed)

                            model = self.efficient_perturb_parameters(model, seed, scaling_factor=1)
                            loss1s.append(self.zo_forward(model, inputs))
                            model = self.efficient_perturb_parameters(model, seed, scaling_factor=-1)
                    d = torch.stack(loss1s) - loss0 # lists of diffed tensors with baseline loss0
                loss1 = loss0 # notation reasons, can change

                # Reuse DPZero gradient clipping for each random direction
                d_clipped = dpzero_clip(d / self.args.zero_order_eps, self.args.dpzero_clip_threshold / math.sqrt(N))
                g = d_clipped.mean(dim=1)
                g = g + torch.randn(N, generator=self.noise_gen).to(g.device) * self.dpzero_gaussian_std  # DP noise

                # FZOO adaptive step size
                g_std = g.std(unbiased=False)
                projected_grads = g / (N * g_std)

                # Apply gradient updates
                if self.fzoo_beta > 0:
                    # ===== MOMENTUM CALCULATION (Nesterov) ==================================================
                    # projected_grads = kappa_i are already clipped + noised + normalized (privatized), so the
                    # momentum below is post-processing: no extra privacy cost, accounting unchanged.
                    #   g      = sum_i kappa_i z_i
                    #   v_new  = beta * v_old + (1 - beta) * g                        (velocity buffer self.fzoo_mom)
                    #   theta -= lr * ((1 - beta) * g + beta * v_new)                 (Nesterov step)
                    if self.zp is not None:
                        self.zp.apply_update_nesterov(projected_grads, self.args.learning_rate,
                                                      self.fzoo_beta, self.fzoo_mom)    # batched path, fused kernel
                    else:
                        from src.zo_parallel import nesterov_update_sequential
                        nesterov_update_sequential(self.named_parameters_to_optim, random_seeds, projected_grads,
                                                   self.args.learning_rate, self.fzoo_beta, self.fzoo_mom)
                    # ========================================================================================
                elif self.zp is not None:
                    self.zp.apply_update(projected_grads, self.args.learning_rate)   # reuses the stored int8 directions
                else:
                    for i in range(N):
                        # resample seeds (no need to store them)
                        torch.manual_seed(random_seeds[i])
                        for name, param in self.named_parameters_to_optim:
                            z = torch.randint(0, 2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype) * 2 - 1
                            # GD Update
                            param.data = param.data - self.args.learning_rate * projected_grads[i] * z

                # logging
                if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                    self.state.global_step == 1 and self.args.logging_first_step
                ):
                    logs = {}
                    logs["loss"] = loss1.mean().item()
                    logs["learning_rate"] = self.args.learning_rate
                    logs["N"] = N
                    logs["momentum"] = self.fzoo_beta
                    logs["global_step"] = self.state.global_step
                    logs["zo_forward_step"] = self.state.zo_forward_step
                    logs["max_steps"] = self.args.max_steps
                    logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                    logs["time"] = time.time() - start_time
                    self.log(logs)
                    logger.info(str(logs))
                
                model.zero_grad()
                self.state.global_step += 1
                self.epoch = epoch + (step + 1) / len(epoch_iterator)
                
                # Debug information
                # print("%.5f, %.5f" % (loss1.item(), loss2.item()))
                # print("Loss: %.10f, projected_grad: %.5f" % (loss1, projected_grad))

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output