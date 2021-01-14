import unittest
import numpy as np
from central_tendency.mean import mean
from central_tendency.median import median
from central_tendency.mode import mode, Mode
from utils.objects import Device, Framework


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
        calculated_mean = mean(observations, framework=Framework.Pytorch, device=Device.CPU)
        self.assertEqual(calculated_mean, 167)

    def test_mean_pytorch_GPU(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_mean = mean(observations, framework=Framework.Pytorch, device=Device.GPU)
        self.assertEqual(calculated_mean, 167)

    def test_mean_tensorflow(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_mean = mean(observations, framework=Framework.Tensorflow, device=Device.GPU)
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

    def test_mode_partitions(self):
        observations = [0, 1, 1, 1, 1, 2, 2, 3, 3, 5, 1, 1, 2, 3, 5, 2, 1, 1, 2, 5]
        x_partitions = Mode(observations, use_pytorch=False, device=None).get_partitions
        expected_list = [[0, 1, 1, 1, 1, 2, 2, 3, 3, 5],
                         [1, 1, 2, 3, 5, 2, 1, 1, 2, 5]]
        self.assertListEqual(x_partitions, expected_list)

    def test_mode_map(self):
        observations = [0, 1, 1, 1, 1, 2, 2, 3, 3, 5, 1, 1, 2, 3, 5, 2, 1, 1, 2, 5]
        x_map = Mode(observations, use_pytorch=False, device=None).get_map
        expected_list = [
            {
                0: 1,
                1: 4,
                2: 2,
                3: 2,
                5: 1
            },
            {
                1: 4,
                2: 3,
                3: 1,
                5: 2
            }]
        self.assertListEqual(x_map, expected_list)

    def test_mode_reduce(self):
        observations = [0, 1, 1, 1, 1, 2, 2, 3, 3, 5, 1, 1, 2, 3, 5, 2, 1, 1, 2, 5]
        x_map = Mode(observations, use_pytorch=False, device=None).get_reduce
        expected_dict = {
            0: 1,
            1: 8,
            2: 5,
            3: 3,
            5: 3
        }
        self.assertDictEqual(x_map, expected_dict)

    def test_mode(self):
        observations = [0, 1, 1, 1, 1, 2, 2, 3, 3, 5, 1, 1, 2, 3, 5, 2, 1, 1, 2, 5]
        mode_list = mode(observations, use_pytorch=False, device=None)
        self.assertListEqual(mode_list, [1])


if __name__ == '__main__':
    unittest.main()
