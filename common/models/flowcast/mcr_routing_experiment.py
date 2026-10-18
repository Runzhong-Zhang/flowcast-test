"""Training-only greedy MCR adaptation; no persistent model parameters/buffers.

Players are existing routing parameter groups, not independent modality encoders.
Auxiliary gradients are computed on the unwrapped model and explicitly averaged
across ranks before being added to already-synchronized task gradients.
"""
from contextlib import contextmanager
from types import MethodType
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def routing_jsd(p_logits, q_logits):
    """Mean token JSD over all experts, before top-k; natural-log units."""
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    return (0.5 * (log_p.exp() * (log_p - log_m)
                   + log_q.exp() * (log_q - log_m)).sum(-1)).mean().clamp_min(0.0)


def greedy_objectives(d_v, d_h):
    # Minimize own negative divergence, maximize the competitor's negative
    # divergence. These must be differentiated against DIFFERENT parameters.
    return -d_v + d_h, -d_h + d_v


def instance_contrastive(h, v, temperature):
    """Paired gate-side features; instance contrastive, not label supervised."""
    h = F.normalize(h.float(), dim=-1)
    v = F.normalize(v.float(), dim=-1)
    logits = h @ v.T / temperature
    labels = torch.arange(h.shape[0], device=h.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class GreedyRoutingMCR:
    def __init__(self, model, config):
        self.model = model
        self.weight = float(config.get('weight', 0.001))
        self.con_weight = float(config.get('contrastive_weight', 0.001))
        self.interval = int(config.get('every_n_steps', 1))
        self.microbatch = int(config.get('micro_batch_size', 8))
        if config.get('divergence', 'routing_jsd') != 'routing_jsd':
            raise ValueError('This experiment requires divergence=routing_jsd')
        self.con_temperature = float(config.get('contrastive_temperature', 0.1))
        self.warmup = int(config.get('warmup_epochs', 0))
        self.ramp = int(config.get('ramp_epochs', 0))
        if int(config.get('k', -1)) != -1:
            raise ValueError('This experiment implements only greedy k=-1')
        if not all(math.isfinite(x) for x in (self.weight, self.con_weight, self.con_temperature)):
            raise ValueError('MCR settings must be finite')
        if min(self.weight, self.con_weight, self.warmup, self.ramp) < 0 or self.interval < 1:
            raise ValueError('Invalid MCR weights/schedule')
        if self.microbatch < 2 or self.con_temperature <= 0:
            raise ValueError('MCR requires microbatch >= 2 and positive temperatures')
        self.layers = [m for m in model.modules() if hasattr(m, '_routing_logits') and hasattr(m, 'vil_value_proj')]
        if not self.layers or getattr(model, 'cmi_routing', False):
            raise ValueError('MCR requires gated cross-modal layers and cmi_weight=0')
        self.v_params, self.h_params = [], []
        for layer in self.layers:
            self.v_params.extend(p for module in (layer.vil_proj, layer.vil_value_proj) for p in module.parameters() if p.requires_grad)
            self.h_params.extend(p for module in (layer.gate_proj, layer.router) for p in module.parameters() if p.requires_grad)
        assert not ({id(p) for p in self.v_params} & {id(p) for p in self.h_params})

    def factor(self, epoch):
        return min(1.0, max(0.0, (epoch - self.warmup + 1) / max(1, self.ramp)))

    def due(self, epoch, batch_index):
        return (self.weight > 0 or self.con_weight > 0) and self.factor(epoch) > 0 and batch_index % self.interval == 0

    @contextmanager
    def intervention(self, permutation, features, divergences):
        """Collect local counterfactual JSD; dispatch uses original logits."""
        originals = []
        batch = permutation.numel()
        try:
            for layer in self.layers:
                had_override = '_routing_logits' in layer.__dict__
                original = layer._routing_logits
                originals.append((layer, original, had_override))

                def routing(module, flat, vil_proj, original=original):
                    logits = original(flat, vil_proj)
                    # Reproject detached conditioning to isolate auxiliary
                    # gradients from upstream hidden states/routing layers.
                    h_local = flat.detach()
                    v_local = module.vil_proj(module._vil_cond.reshape(flat.shape[0], -1).detach())
                    if features is not None:
                        # Existing gate projection's H block: no new head and
                        # no CON gradient into transformer hidden states.
                        h = F.linear(flat.detach(), module.gate_proj.weight[:, :module.units], module.gate_proj.bias)
                        h = h.reshape(batch, -1, flat.shape[-1]).mean(1)
                        v = module.vil_value_proj(v_local).reshape(batch, -1, flat.shape[-1]).mean(1)
                        features.append((h, v))
                    if self.weight:
                        v_shuffle = v_local.reshape(batch, -1, v_local.shape[-1])[permutation].reshape_as(v_local)
                        h_shuffle = h_local.reshape(batch, -1, h_local.shape[-1])[permutation].reshape_as(h_local)
                        base = original(h_local, v_local)
                        divergences.append((routing_jsd(base, original(h_local, v_shuffle)),
                                            routing_jsd(base, original(h_shuffle, v_local))))
                    return logits

                layer._routing_logits = MethodType(routing, layer)
            yield
        finally:
            for layer, original, had_override in originals:
                if had_override:
                    layer._routing_logits = original
                else:
                    del layer._routing_logits

    def add_gradients(self, z_vil, lightning, future_vil, t, x_t, epoch, scale=1.0):
        """Add scaled auxiliary gradients BEFORE GradScaler.unscale_.

        The main task backward must be complete first. No DDP forward/backward
        occurs here; autograd.grad outputs are manually reduced in fixed order.
        """
        batch = min(self.microbatch, lightning.shape[0])
        if batch < 2:
            raise ValueError('MCR local batch must contain at least two examples')
        z_vil, lightning = z_vil[:batch].detach(), lightning[:batch].detach()
        t, x_t = t[:batch].detach(), x_t[:batch].detach()
        future_vil = None if future_vil is None else future_vil[:batch].detach()
        states = [(m, m.training) for m in self.model.modules()]
        params = list(self.model.parameters())
        flags = [p.requires_grad for p in params]
        owned = {id(p) for p in self.v_params + self.h_params}
        # Save injection/cache attributes so no auxiliary graph survives a step.
        cache_names = ('_vil_cond', '_aux_loss_for_bp', '_cmi_loss_for_bp', '_cmi_permutation')
        caches = [(m, {n: getattr(m, n) for n in cache_names if hasattr(m, n)}) for m in self.layers]
        device = lightning.device
        cuda_devices = [device.index] if device.type == 'cuda' else []
        try:
            # Eval disables dropout and checkpoint recomputation of mutable
            # per-forward router conditions. It does not disable autograd.
            self.model.eval()
            for p in params:
                p.requires_grad_(id(p) in owned)
            with torch.random.fork_rng(devices=cuda_devices):
                permutation = (torch.arange(batch, device=device) + torch.randint(1, batch, (), device=device)) % batch
                features = [] if self.con_weight else None
                divergences = []
                with self.intervention(permutation, features, divergences):
                    self.model.forward_from_encoded(t, x_t, z_vil, lightning, z_vil_future_pred=future_vil)
                zero = self.v_params[0].sum() * 0
                dv = torch.stack([d[0] for d in divergences]).mean() if divergences else zero
                dh = torch.stack([d[1] for d in divergences]).mean() if divergences else zero
                con = (torch.stack([instance_contrastive(h, v, self.con_temperature) for h, v in features]).mean()
                       if features else zero)
                lv, lh = greedy_objectives(dv, dh)
                factor = self.factor(epoch)
                gv = torch.autograd.grad(scale * factor * (self.weight * lv + self.con_weight * con), self.v_params, retain_graph=True, allow_unused=True)
                gh = torch.autograd.grad(scale * factor * (self.weight * lh + self.con_weight * con), self.h_params, allow_unused=True)
                logs = dict(routing_jsd_vil=float(dv.detach()), routing_jsd_hidden=float(dh.detach()), contrastive=float(con.detach()), mcr_factor=factor)
            task_sq = aux_sq = dot = 0.0
            for p, grad in zip(self.v_params + self.h_params, gv + gh):
                grad = torch.zeros_like(p) if grad is None else grad.detach()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(grad)
                    grad.div_(dist.get_world_size())
                task = torch.zeros_like(p) if p.grad is None else p.grad
                task_sq += float((task.float() / scale).square().sum())
                aux_sq += float((grad.float() / scale).square().sum())
                dot += float(((task.float() / scale) * (grad.float() / scale)).sum())
                if p.grad is None:
                    p.grad = grad
                else:
                    p.grad.add_(grad)
            logs.update(aux_grad_norm=math.sqrt(aux_sq), task_router_grad_norm=math.sqrt(task_sq),
                        grad_cosine=dot / max(1e-12, math.sqrt(task_sq * aux_sq)))
            return logs
        finally:
            for p, flag in zip(params, flags):
                p.requires_grad_(flag)
            for module, training in states:
                module.training = training
            for module, cache in caches:
                for name, value in cache.items():
                    setattr(module, name, value)
