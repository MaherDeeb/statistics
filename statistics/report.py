class Report:
    def __init__(self, is_test=False):
        self.statistics_report = None
        if is_test:
            self.statistics_report = "This is a test report"

    @property
    def get_statistics(self):
        return self.statistics_report
