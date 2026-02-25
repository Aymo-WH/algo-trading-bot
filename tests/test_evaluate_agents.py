import sys
import os
import pytest
from datetime import datetime, timedelta

# Ensure the parent directory is in the path to import evaluate_agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate_agents import calculate_cagr

class TestCalculateCAGR:

    def test_positive_growth(self):
        """Test CAGR with positive growth over 1 year."""
        start_val = 100.0
        end_val = 110.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1) # 365 days

        # calculate_cagr uses 365.25 as year length
        # expected = (110/100)^(365.25/365) - 1
        # years = days / 365.25
        # years = 365 / 365.25 = 0.9993155
        # (1.1)^(1/0.9993155) - 1

        cagr = calculate_cagr(start_val, end_val, start_date, end_date)
        assert cagr > 0.09
        assert cagr < 0.11

    def test_exact_one_year_growth(self):
        """Test CAGR with exact one year duration based on 365.25 days."""
        start_val = 100.0
        end_val = 110.0
        start_date = datetime(2023, 1, 1)
        # 365.25 days is tricky with datetime since days are integers.
        # So we can't get exactly 1.0 years unless we mock days difference or use multiple years (4 years = 1461 days).

        # Let's test 4 years = 1461 days. 1461 / 365.25 = 4.0
        end_date = start_date + timedelta(days=1461)

        # If we grew from 100 to 146.41 (1.1^4 * 100)
        end_val = 100.0 * (1.1 ** 4)

        cagr = calculate_cagr(start_val, end_val, start_date, end_date)
        assert abs(cagr - 0.10) < 1e-6

    def test_negative_growth(self):
        """Test CAGR with loss over 1 year."""
        start_val = 100.0
        end_val = 90.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)

        cagr = calculate_cagr(start_val, end_val, start_date, end_date)
        assert cagr < 0.0

    def test_short_period(self):
        """Test CAGR for a short period (e.g., 30 days)."""
        start_val = 100.0
        end_val = 101.0 # 1% gain in 30 days
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=30)

        # years = 30 / 365.25 ~ 0.082
        # (1.01)^(1/0.082) - 1 ~ (1.01)^12.175 - 1 ~ 1.129

        cagr = calculate_cagr(start_val, end_val, start_date, end_date)
        assert cagr > 0.10 # Should be annualized significantly

    def test_zero_days(self):
        """Test behavior when start_date equals end_date."""
        start_val = 100.0
        end_val = 110.0
        start_date = datetime(2023, 1, 1)
        end_date = start_date

        assert calculate_cagr(start_val, end_val, start_date, end_date) == 0.0

    def test_negative_days(self):
        """Test behavior when end_date is before start_date."""
        start_val = 100.0
        end_val = 110.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2022, 1, 1)

        assert calculate_cagr(start_val, end_val, start_date, end_date) == 0.0

    def test_zero_start_value(self):
        """Test behavior when start_value is 0."""
        start_val = 0.0
        end_val = 100.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)

        assert calculate_cagr(start_val, end_val, start_date, end_date) == 0.0

    def test_negative_start_value(self):
        """Test behavior when start_value is negative."""
        start_val = -100.0
        end_val = 100.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)

        assert calculate_cagr(start_val, end_val, start_date, end_date) == 0.0

    def test_zero_end_value(self):
        """Test behavior when end_value is 0 (total loss)."""
        start_val = 100.0
        end_val = 0.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)

        assert calculate_cagr(start_val, end_val, start_date, end_date) == -1.0

    def test_negative_end_value(self):
        """Test behavior when end_value is negative."""
        start_val = 100.0
        end_val = -10.0
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)

        assert calculate_cagr(start_val, end_val, start_date, end_date) == -1.0
