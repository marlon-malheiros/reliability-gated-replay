# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello,
# Davide Abati, Simone Calderara.
# PNN-DER++ extension added for the PNN consolidation project.

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


def _normalize(t: torch.Tensor) -> torch.Tensor:
    lo = t.min()
    hi = t.max()
    return (t - lo) / (hi - lo + 1e-12)


class PnnDerpp(ContinualModel):
    """DER++ with an autonomous P-weighted consolidation anchor.

    This keeps Mammoth's DER++ replay/distillation path intact and adds only:
    1. gradient/magnitude-derived parameter importance;
    2. relative-plateau closure detection at epoch boundaries;
    3. a P-weighted quadratic anchor to task-end snapshots.
    """

    NAME = 'pnn_derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='DER++ stored-logit distillation weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='DER++ replay cross-entropy weight.')
        parser.add_argument('--pnn_lambda', type=float, default=0.3,
                            help='Weight of the P-weighted quadratic anchor.')
        parser.add_argument('--pnn_anchor_norm', type=str, default='param_mean',
                            choices=['param_mean', 'p_weighted_mean', 'sum'],
                            help='Anchor normalization: old total-parameter mean, P-weighted mean, or unnormalized sum.')
        parser.add_argument('--pnn_maturation_alpha', type=float, default=0.1,
                            help='Per-epoch maturation step for P.')
        parser.add_argument('--pnn_importance_decay', type=float, default=0.9,
                            help='EMA decay for gradient-based importance.')
        parser.add_argument('--pnn_plateau_window', type=int, default=3,
                            help='Epoch window for relative plateau closure.')
        parser.add_argument('--pnn_closure_ema_alpha', type=float, default=0.0,
                            help='EMA smoothing for closure loss. 0 disables smoothing.')
        parser.add_argument('--pnn_min_mature_epoch', type=int, default=0,
                            help='Earliest 1-indexed epoch where maturation may update P.')
        parser.add_argument('--pnn_min_mature_epoch_frac', type=float, default=0.0,
                            help='Earliest fraction of task epochs where maturation may update P.')
        parser.add_argument('--pnn_rel_improve_eps', type=float, default=0.02,
                            help='Relative improvement below this counts as plateau.')
        parser.add_argument('--pnn_norm_var_eps', type=float, default=0.02,
                            help='Normalized loss variance below this counts as stable.')
        parser.add_argument('--pnn_closure_fire_threshold', type=float, default=0.5,
                            help='Diagnostic threshold for logging closure events.')
        parser.add_argument('--pnn_eval_closure_loss', type=int, default=1,
                            help='Use validation/test loss for closure when available.')
        parser.add_argument('--pnn_include_classifier', type=int, default=1,
                            help='Include classifier parameters in P and anchor.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        self._named: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if not self.args.pnn_include_classifier and self._is_classifier_param(name):
                continue
            self._named.append((name, param))

        self.P: Dict[str, torch.Tensor] = {n: torch.zeros_like(p, device=self.device) for n, p in self._named}
        self.grad_ema: Dict[str, torch.Tensor] = {n: torch.zeros_like(p, device=self.device) for n, p in self._named}
        self.grad2_ema: Dict[str, torch.Tensor] = {n: torch.zeros_like(p, device=self.device) for n, p in self._named}
        self.anchor_star: Dict[str, torch.Tensor] = {}
        self.anchor_updates = 0
        self.loss_history: List[float] = []
        self.smoothed_loss: float | None = None
        self.train_epoch_losses: List[float] = []
        self.epoch_components: Dict[str, List[float]] = {}
        self.current_closure_signal = 0.0
        self.closure_events: List[dict] = []

    @staticmethod
    def _is_classifier_param(name: str) -> bool:
        lname = name.lower()
        return any(token in lname for token in ('classifier', 'classif', 'fc', 'linear'))

    def begin_task(self, dataset) -> None:
        self.loss_history = []
        self.smoothed_loss = None
        self.current_closure_signal = 0.0

    def begin_epoch(self, epoch: int, dataset) -> None:
        self.train_epoch_losses = []
        self.epoch_components = {
            'stream': [],
            'der_mse': [],
            'der_ce': [],
            'anchor': [],
            'total': [],
        }

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss_stream = self.loss(outputs, labels)
        loss = loss_stream
        loss_mse = loss_stream.new_zeros(())
        loss_ce = loss_stream.new_zeros(())
        anchor = None

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss = loss + loss_mse

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss = loss + loss_ce

        anchor = self.anchor_loss()
        if anchor is not None:
            loss = loss + anchor

        loss.backward()
        self._update_importance()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
        self.train_epoch_losses.append(float(loss.detach().item()))
        self.epoch_components['stream'].append(float(loss_stream.detach().item()))
        self.epoch_components['der_mse'].append(float(loss_mse.detach().item()))
        self.epoch_components['der_ce'].append(float(loss_ce.detach().item()))
        self.epoch_components['anchor'].append(float(anchor.detach().item()) if anchor is not None else 0.0)
        self.epoch_components['total'].append(float(loss.detach().item()))
        return loss.item()

    def end_epoch(self, epoch: int, dataset) -> None:
        raw_loss_value = self._closure_loss(dataset)
        if raw_loss_value is None:
            return
        loss_value = self._smooth_closure_loss(float(raw_loss_value))
        self.loss_history.append(float(loss_value))
        raw_signal, rel_improve, norm_var = self._relative_plateau_signal()
        maturity_allowed = self._maturity_allowed(epoch)
        signal = raw_signal if maturity_allowed else 0.0
        self.current_closure_signal = signal
        self._mature(signal)
        mean_p = self.mean_P()
        comps = {k: (float(np.mean(v)) if v else 0.0) for k, v in self.epoch_components.items()}
        anchor_over_stream = comps['anchor'] / max(comps['stream'], 1e-12)
        msg = (
            f"[PNN] task={self.current_task} epoch={epoch + 1} "
            f"closure_signal={signal:.4f} raw_closure_signal={raw_signal:.4f} "
            f"maturity_allowed={int(maturity_allowed)} rel_improve={rel_improve:.4f} "
            f"norm_var={norm_var:.4f} loss={loss_value:.6f} raw_loss={raw_loss_value:.6f} "
            f"mean_P={mean_p:.6f} "
            f"stream_loss={comps['stream']:.6f} der_mse={comps['der_mse']:.6f} "
            f"der_ce={comps['der_ce']:.6f} anchor_loss={comps['anchor']:.6f} "
            f"anchor_over_stream={anchor_over_stream:.6e} anchor_norm={self.args.pnn_anchor_norm}"
        )
        logging.info(msg)
        print(msg, flush=True)
        if signal >= self.args.pnn_closure_fire_threshold:
            event = {
                'task': int(self.current_task),
                'epoch': int(epoch + 1),
                'closure_signal': float(signal),
                'raw_closure_signal': float(raw_signal),
                'loss': float(loss_value),
                'raw_loss': float(raw_loss_value),
                'mean_P': float(mean_p),
            }
            self.closure_events.append(event)
            logging.info(f"[PNN] closure_fire {event}")

    def end_task(self, dataset) -> None:
        self.snapshot_anchor()
        logging.info(
            f"[PNN] task_end task={self.current_task} mean_P={self.mean_P():.6f} "
            f"anchor_updates={self.anchor_updates}"
        )

    def _closure_loss(self, dataset) -> float | None:
        if self.args.pnn_eval_closure_loss:
            try:
                _, _, loss = dataset.evaluate(self, dataset, last=True, return_loss=True)
                return float(loss)
            except Exception as e:
                logging.warning(f"[PNN] validation closure loss failed; using train loss. {e}")
        if not self.train_epoch_losses:
            return None
        return float(np.mean(self.train_epoch_losses))

    def _smooth_closure_loss(self, loss_value: float) -> float:
        alpha = float(self.args.pnn_closure_ema_alpha)
        if alpha <= 0:
            return loss_value
        alpha = min(max(alpha, 0.0), 0.999)
        if self.smoothed_loss is None:
            self.smoothed_loss = loss_value
        else:
            self.smoothed_loss = alpha * self.smoothed_loss + (1.0 - alpha) * loss_value
        return float(self.smoothed_loss)

    def _maturity_allowed(self, epoch: int) -> bool:
        by_epoch = max(0, int(self.args.pnn_min_mature_epoch))
        by_frac = int(np.ceil(float(self.args.pnn_min_mature_epoch_frac) * float(self.args.n_epochs)))
        first_epoch = max(1, by_epoch, by_frac)
        return (epoch + 1) >= first_epoch

    def _relative_plateau_signal(self) -> Tuple[float, float, float]:
        w = max(1, int(self.args.pnn_plateau_window))
        if len(self.loss_history) < w + 1:
            return 0.0, float('inf'), float('inf')
        prev = np.asarray(self.loss_history[-w - 1:-1], dtype=np.float64)
        cur = float(self.loss_history[-1])
        prev_mean = float(prev.mean())
        rel_improve = (prev_mean - cur) / max(abs(prev_mean), 1e-12)
        recent = np.asarray(self.loss_history[-w:], dtype=np.float64)
        norm_var = float(recent.std() / max(abs(recent.mean()), 1e-12))

        improve_score = np.clip(
            (self.args.pnn_rel_improve_eps - rel_improve) / max(self.args.pnn_rel_improve_eps, 1e-12),
            0.0,
            1.0,
        )
        var_score = np.clip(
            (self.args.pnn_norm_var_eps - norm_var) / max(self.args.pnn_norm_var_eps, 1e-12),
            0.0,
            1.0,
        )
        return float(improve_score * var_score), float(rel_improve), float(norm_var)

    def _update_importance(self) -> None:
        decay = float(self.args.pnn_importance_decay)
        for n, p in self._named:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            self.grad_ema[n].mul_(decay).add_(grad, alpha=1.0 - decay)
            self.grad2_ema[n].mul_(decay).add_(grad * grad, alpha=1.0 - decay)

    def _importance(self) -> Dict[str, torch.Tensor]:
        out = {}
        for n, p in self._named:
            mag = _normalize(p.detach().abs())
            grad = _normalize(self.grad_ema[n].abs())
            fisher = _normalize(self.grad2_ema[n])
            out[n] = ((mag + grad + fisher) / 3.0).detach()
        return out

    def _mature(self, closure_signal: float) -> None:
        if closure_signal <= 0:
            return
        imp = self._importance()
        step = float(self.args.pnn_maturation_alpha) * float(closure_signal)
        for n, _ in self._named:
            self.P[n] = (self.P[n] + step * imp[n]).clamp_(0.0, 1.0)

    def snapshot_anchor(self) -> None:
        self.anchor_star = {n: p.detach().clone() for n, p in self._named}
        self.anchor_updates += 1

    def anchor_loss(self):
        if not self.anchor_star:
            return None
        loss = None
        param_count = 0
        p_sum = None
        for n, p in self._named:
            weights = self.P[n].detach()
            term = (weights * (p - self.anchor_star[n]).pow(2)).sum()
            loss = term if loss is None else loss + term
            p_term = weights.sum()
            p_sum = p_term if p_sum is None else p_sum + p_term
            param_count += p.numel()
        if loss is None:
            return None
        norm = self.args.pnn_anchor_norm
        if norm == 'param_mean':
            loss = loss / max(param_count, 1)
        elif norm == 'p_weighted_mean':
            loss = loss / p_sum.clamp_min(1e-12)
        elif norm == 'sum':
            pass
        else:
            raise ValueError(f'Unknown pnn_anchor_norm={norm}')
        return 0.5 * float(self.args.pnn_lambda) * loss

    def mean_P(self) -> float:
        if not self.P:
            return 0.0
        return float(torch.cat([p.detach().reshape(-1) for p in self.P.values()]).mean().item())
