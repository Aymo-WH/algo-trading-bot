import unittest
from time_utils import get_seconds_in_a_day, get_seconds_in_a_week

class TestTimeUtils(unittest.TestCase):
    def test_get_seconds_in_a_day(self):
        """Tests the calculation of seconds in a day."""
        self.assertEqual(get_seconds_in_a_day(), 86400)

    def test_get_seconds_in_a_week(self):
        """Tests the calculation of seconds in a week."""
        self.assertEqual(get_seconds_in_a_week(), 604800)

if __name__ == '__main__':
    unittest.main()
