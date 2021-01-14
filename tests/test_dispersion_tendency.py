import unittest
from dispersion_tendency.range import data_range
from utils.objects import Device


class MyTestCase(unittest.TestCase):
    def test_data_range_numpy(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_range = data_range(observations)
        self.assertEqual(calculated_range, 340)

    def test_data_range_pytorch_cpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_range = data_range(observations, use_pytorch=True, device=Device.CPU)
        self.assertEqual(calculated_range, 340)

    def test_data_range_pytorch_gpu(self):
        observations = [0, 340, 70, 140, 200, 180, 210, 150, 100, 130, 140, 180, 190, 160, 290, 50, 220, 180, 200, 210]
        calculated_range = data_range(observations, use_pytorch=True, device=Device.GPU)
        self.assertEqual(calculated_range, 340)


if __name__ == '__main__':
    unittest.main()
