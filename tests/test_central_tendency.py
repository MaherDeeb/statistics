import unittest
import numpy as np
from central_tendency.mean import mean
from utils.objects import Device


class MyTestCase(unittest.TestCase):
    def test_mean_list(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_mean = mean(observations)
        self.assertEqual(calculated_mean, 167)

    def test_mean_array(self):
        observations = np.array([0, 340, 70, 140, 200, 180, 210, 150, 100, 130,
                                 140, 180, 190, 160, 290, 50, 220, 180, 200, 210])
        calculated_mean = mean(observations)
        self.assertEqual(calculated_mean, 167)

    def test_mean_pytorch_cpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_mean = mean(observations, use_pytorch=True, device=Device.CPU)
        self.assertEqual(calculated_mean, 167)

    def test_mean_pytorch_GPU(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_mean = mean(observations, use_pytorch=True, device=Device.GPU)
        self.assertEqual(calculated_mean, 167)


if __name__ == '__main__':
    unittest.main()
