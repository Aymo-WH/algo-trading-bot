import unittest
import numpy as np
import pandas as pd
from core.pbo_validator import PBOValidator

class TestPBOValidator(unittest.TestCase):

    def test_calculate_pbo_deterministic(self):
        """
        Test calculate_pbo with a small, deterministic matrix.
        We have 4 timesteps (S=4 partitions, so each submatrix is 1 timestep) and 2 trials.

        M = [
            [ 1, -1],  # t=0
            [ 1, -1],  # t=1
            [-1,  1],  # t=2
            [-1,  1]   # t=3
        ]

        Submatrices (1x2 each):
        m0 = [ 1, -1]
        m1 = [ 1, -1]
        m2 = [-1,  1]
        m3 = [-1,  1]

        S=4, combinations of size 2 = 6 combinations.

        Comb 1: J = (0, 1) -> sum=[ 2, -2], OOS J_bar = (2, 3) -> sum=[-2,  2]
           train mean = [1, -1], train std = [0, 0] -> Sharpe = [1/1e-8, -1/1e-8] -> opt_idx = 0
           test mean = [-1, 1], test std = [0, 0] -> Sharpe = [-1/1e-8, 1/1e-8]
           OOS ranks: [-large, +large] -> ranks = [1, 2]
           opt_idx = 0, rank = 1
           omega_bar = 1 / (2+1) = 1/3
           logit = ln((1/3) / (2/3)) = ln(0.5) = -0.69314718

        Comb 2: J = (0, 2) -> sum=[ 0,  0], OOS J_bar = (1, 3) -> sum=[ 0,  0]
           train mean = [0, 0], train std = [1, 1] -> opt_idx = 0 (first max)
           test mean = [0, 0], test std = [1, 1] -> Sharpe = [0, 0]
           OOS ranks: [1.5, 1.5]
           opt_idx = 0, rank = 1.5
           omega_bar = 1.5 / 3 = 0.5
           logit = ln(0.5 / 0.5) = 0

        Comb 3: J = (0, 3) -> sum=[ 0,  0], OOS J_bar = (1, 2) -> sum=[ 0,  0]
           train mean = [0, 0], test mean = [0, 0]
           logit = 0

        Comb 4: J = (1, 2) -> sum=[ 0,  0], OOS J_bar = (0, 3) -> sum=[ 0,  0]
           logit = 0

        Comb 5: J = (1, 3) -> sum=[ 0,  0], OOS J_bar = (0, 2) -> sum=[ 0,  0]
           logit = 0

        Comb 6: J = (2, 3) -> sum=[-2,  2], OOS J_bar = (0, 1) -> sum=[ 2, -2]
           train mean = [-1, 1] -> opt_idx = 1
           test mean = [1, -1], OOS ranks = [2, 1]
           opt_idx = 1, rank = 1
           omega_bar = 1 / 3
           logit = ln(0.5) = -0.69314718

        Total logits = [-0.693, 0, 0, 0, 0, -0.693]
        PBO = 2 / 6 = 1/3 = 0.3333333
        """
        M = np.array([
            [ 1, -1],
            [ 1, -1],
            [-1,  1],
            [-1,  1]
        ])
        df = pd.DataFrame(M)

        validator = PBOValidator(df, num_partitions=4)
        pbo, logits = validator.calculate_pbo()

        self.assertAlmostEqual(pbo, 0.3333333333333333)
        self.assertEqual(len(logits), 6)

        expected_logits = np.array([np.log(0.5), 0, 0, 0, 0, np.log(0.5)])
        np.testing.assert_allclose(logits, expected_logits, atol=1e-7)

    def test_calculate_pbo_perfect_overfit(self):
        """
        Test a scenario where one model performs well IS but terribly OOS,
        and another performs consistently.
        """
        # 4 timesteps, 2 models
        # Model 1 is volatile/noisy: great in first half, terrible in second half
        # Model 2 is consistent: decent everywhere
        M = np.array([
            [ 2,  1],
            [ 2,  1],
            [-2,  1],
            [-2,  1]
        ])
        df = pd.DataFrame(M)

        validator = PBOValidator(df, num_partitions=4)
        pbo, logits = validator.calculate_pbo()

        # We verify it runs and pbo is within [0, 1]
        self.assertTrue(0 <= pbo <= 1)
        self.assertEqual(len(logits), 6)

    def test_calculate_pbo_edge_cases(self):
        """
        Test behavior with 0 variance (constant returns) to ensure the `1e-8` in the denominator
        correctly avoids NaNs or division by zero errors.
        """
        # All models have exactly the same returns everywhere
        M = np.array([
            [ 1,  1],
            [ 1,  1],
            [ 1,  1],
            [ 1,  1]
        ])
        df = pd.DataFrame(M)

        validator = PBOValidator(df, num_partitions=4)
        pbo, logits = validator.calculate_pbo()

        # In this case all models tie so rankdata will give them all 1.5 rank
        # omega_bar will be 0.5, logit will be 0.
        self.assertAlmostEqual(pbo, 0.0) # no logits < 0
        expected_logits = np.zeros(6)
        np.testing.assert_allclose(logits, expected_logits, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
