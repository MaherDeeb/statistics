import unittest
import numpy as np
from central_tendency.mean import mean
from central_tendency.median import median
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

    def test_median_list_even(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 190, 200, 210]
        calculated_median = median(observations)
        self.assertEqual(calculated_median, 185)

    def test_median_list_odd(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 190, 200]
        calculated_median = median(observations)
        self.assertEqual(calculated_median, 180)

    def test_median_list_odd_ex_2(self):
        observations = [0.3, 0.4, 0.8, 1.4, 1.8, 2.1, 5.9, 11.6, 16.9]
        calculated_median = median(observations)
        self.assertEqual(calculated_median, 1.8)

    def test_median_list_even_pytorch_cpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 190, 200, 210]
        calculated_median = median(observations, use_pytorch=True, device=Device.CPU)
        self.assertEqual(calculated_median, 185)

    def test_median_list_odd_pytorch_cpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 190, 200]
        calculated_median = median(observations, use_pytorch=True, device=Device.CPU)
        self.assertEqual(calculated_median, 180)

    def test_median_list_odd_ex_2_pytorch_cpu(self):
        observations = [0.3, 0.4, 0.8, 1.4, 1.8, 2.1, 5.9, 11.6, 16.9]
        calculated_median = median(observations, use_pytorch=True, device=Device.CPU)
        self.assertEqual(calculated_median, 1.8)

    def test_median_list_even_pytorch_gpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 190, 200, 210]
        calculated_median = median(observations, use_pytorch=True, device=Device.GPU)
        self.assertEqual(calculated_median, 185)

    def test_median_list_odd_pytorch_gpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 190, 200]
        calculated_median = median(observations, use_pytorch=True, device=Device.GPU)
        self.assertEqual(calculated_median, 180)

    def test_median_list_odd_ex_2_pytorch_gpu(self):
        observations = [0.3, 0.4, 0.8, 1.4, 1.8, 2.1, 5.9, 11.6, 16.9]
        calculated_median = median(observations, use_pytorch=True, device=Device.GPU)
        self.assertEqual(calculated_median, 1.8)


if __name__ == '__main__':
    unittest.main()
