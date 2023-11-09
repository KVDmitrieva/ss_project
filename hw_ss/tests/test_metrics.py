import torch
import unittest

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_ss.metric.utils import calc_si_sdr


class TestMetrics(unittest.TestCase):
    def test_si_sdr_metric(self):
        torch.manual_seed(1)

        target = torch.randn(5)
        preds = torch.randn(5)

        torch_si_sdr = ScaleInvariantSignalDistortionRatio()

        torch_result = torch_si_sdr(preds, target)
        custom_result = calc_si_sdr(preds, target)

        self.assertAlmostEqual(torch_result.item(), custom_result.item(), places=2)

    def test_batch_si_sdr_metric(self):
        torch.manual_seed(1)

        target = torch.randn(4, 1000)
        preds = torch.randn(4, 1000)

        torch_si_sdr = ScaleInvariantSignalDistortionRatio()

        torch_result = torch_si_sdr(preds, target)
        custom_result = calc_si_sdr(preds, target)

        self.assertAlmostEqual(torch_result.item(), custom_result.item(), places=2)
