import unittest
import torch

from src.model import Detector


class MyTest(unittest.TestCase):
    @torch.inference_mode()
    def test_efficientnet(self):
        model = Detector('efficientnet')
        x = torch.rand(2, 3, 380, 380)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))

    @torch.inference_mode()
    def test_farl(self):
        model = Detector('farl')
        x = torch.rand(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
