import unittest
from statistics.report import Report


class MyTestCase(unittest.TestCase):
    def test_something(self):
        reporting = Report(is_test=True)
        self.assertEqual(reporting.statistics_report, "This is a test report")


if __name__ == '__main__':
    unittest.main()
