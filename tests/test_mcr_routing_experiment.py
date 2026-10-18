"""Focused CPU checks: python -m unittest discover -s tests -p test_mcr_routing_experiment.py -v."""
import math
import unittest

import torch
from torch import nn

from common.models.flowcast.mcr_routing_experiment import GreedyRoutingMCR, routing_jsd


class ToyGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = 3
        self.vil_proj = nn.Linear(2, 3)
        self.vil_value_proj = nn.Linear(3, 3)
        self.gate_proj = nn.Linear(6, 3)
        self.router = nn.Linear(3, 4)
        self._vil_cond = None

    def _routing_logits(self, h, v):
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, v], -1)))
        return self.router(h + gate * self.vil_value_proj(v))


class RoutingJSDTests(unittest.TestCase):
    def test_jsd_properties_and_gradients(self):
        p = torch.tensor([[5., -5., 0.]], requires_grad=True)
        q = torch.tensor([[-5., 5., 0.]], requires_grad=True)
        d = routing_jsd(p, q)
        self.assertAlmostEqual(float(routing_jsd(p, p)), 0., places=6)
        torch.testing.assert_close(d, routing_jsd(q, p))
        self.assertGreater(float(d), 0.)
        self.assertLessEqual(float(d), math.log(2.) + 1e-6)
        d.backward()
        for grad in (p.grad, q.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)

    def test_local_interventions_keep_dispatch_and_isolate_inputs(self):
        torch.manual_seed(7)
        layer = ToyGate()
        model = nn.Sequential(layer)
        mcr = GreedyRoutingMCR(model, {})
        hidden = torch.randn(6, 3, requires_grad=True)
        raw_vil = torch.randn(6, 2, requires_grad=True)
        layer._vil_cond = raw_vil
        v = layer.vil_proj(raw_vil)
        expected = layer._routing_logits(hidden, v)
        features, divergences = [], []
        with mcr.intervention(torch.tensor([1, 0]), features, divergences):
            actual = layer._routing_logits(hidden, v)
        torch.testing.assert_close(actual, expected)
        self.assertNotIn('_routing_logits', layer.__dict__)
        self.assertEqual(len(divergences), 1)
        dv, dh = divergences[0]
        grads = torch.autograd.grad(-dv + dh, mcr.v_params, retain_graph=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        grads = torch.autograd.grad(-dh + dv, mcr.h_params, retain_graph=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        self.assertEqual(torch.autograd.grad(dv + dh, (hidden, raw_vil), allow_unused=True), (None, None))

    def test_exception_restores_router(self):
        layer = ToyGate()
        mcr = GreedyRoutingMCR(nn.Sequential(layer), {})
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with mcr.intervention(torch.tensor([1, 0]), [], []):
                raise RuntimeError('test failure')
        self.assertNotIn('_routing_logits', layer.__dict__)


if __name__ == '__main__':
    unittest.main()
