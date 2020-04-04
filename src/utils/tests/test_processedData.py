from unittest import TestCase
import pandas as pd

from src.utils.processed_data_class import ProcessedData


class TestProcessedData(TestCase, ProcessedData):
    def setUp(self):
        super().__init__()
        self.test_timeser = pd.Series([pd.to_datetime('2020-03-14 01:00:50'),
                                       pd.to_datetime('2020-04-14 01:00:00')])
        self.test_date = pd.Series([0, 31])
        self.test_time = pd.Series([3650, 3600])

    def test_timeser_to_date_and_time(self):
        date, time = ProcessedData.timeser_to_date_and_time(self.test_timeser)
        pd.testing.assert_series_equal(date, self.test_date)
        pd.testing.assert_series_equal(time, self.test_time)

    def test_date_and_time_to_timeser(self):
        timeser = ProcessedData.date_and_time_to_timeser(self.test_date, self.test_time)
        pd.testing.assert_series_equal(timeser, self.test_timeser)
