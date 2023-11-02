import torch
import unittest

from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio

from hw_ss.metric.sdr_metric import SDRMetric
from hw_ss.metric.si_sdr_metric import SISDRMetric


class TestMetrics(unittest.TestCase):
    def test_sdr_metric(self):
        torch.manual_seed(1)

        target = torch.randn(1000)
        preds = torch.randn(1000)

        torch_sdr = SignalDistortionRatio()
        custom_sdr = SDRMetric()

        torch_result = torch_sdr(preds, target)
        custom_result = custom_sdr(preds, target)

        self.assertAlmostEqual(custom_result.item(), torch_result.item(), places=2)

    def test_si_sdr_metric(self):
        torch.manual_seed(1)

        target = torch.randn(5)
        preds = torch.randn(5)

        torch_si_sdr = ScaleInvariantSignalDistortionRatio()
        custom_si_sdr = SISDRMetric()

        torch_result = torch_si_sdr(preds, target)
        custom_result = custom_si_sdr(preds, target)

        self.assertAlmostEqual(torch_result.item(), custom_result.item(), places=2)

    def test_batch_si_sdr_metric(self):
        torch.manual_seed(1)

        target = torch.randn(4, 1000)
        preds = torch.randn(4, 1000)

        torch_si_sdr = ScaleInvariantSignalDistortionRatio()
        custom_si_sdr = SISDRMetric()

        torch_result = torch_si_sdr(preds, target)
        custom_result = custom_si_sdr(preds, target)

        self.assertAlmostEqual(torch_result.item(), custom_result.item(), places=2)
