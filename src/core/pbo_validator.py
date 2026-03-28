import numpy as np
import pandas as pd
import itertools
from scipy.stats import rankdata

class PBOValidator:
    """
    Implements Combinatorially Symmetric Cross-Validation (CSCV) to compute the
    Probability of Backtest Overfitting (PBO).

    This technique estimates the likelihood that a model's optimal performance in-sample
    was purely due to noise, rather than true signal, by generating thousands of
    recombinations of the performance matrix and evaluating their corresponding out-of-sample ranks.
    """
    def __init__(self, performance_matrix, num_partitions=16):
        """
        Initializes the PBO Validator.

        Args:
            performance_matrix (np.ndarray): Shape (Trials, Timesteps), where each row
                is the historical performance of a tested hyperparameter set.
            num_partitions (int): Number of chunks (S) to split the timeline into.
                Defaults to 16, producing 12,870 combinations.
        """
        # performance_matrix is a TxN pandas DataFrame (T=timesteps, N=model trials)
        self.M = performance_matrix
        self.S = num_partitions
        self.T, self.N = self.M.shape

    def calculate_pbo(self):
        """
        Calculates the Probability of Backtest Overfitting.

        Splits the performance matrix into `S` equal parts, forms `S/2` training
        recombinations (IS) and corresponding testing recombinations (OOS). Finds the
        trial that performs best IS, and evaluates its percentile rank OOS. The PBO
        is the percentage of recombinations where the IS winner performs worse than
        the median (rank < 0.5) in the OOS set.

        Returns:
            float: The calculated PBO value (e.g., 0.15 means 15% probability).
        """
        # Step 1: Partition into S sub-matrices
        submatrices = np.array_split(self.M.values, self.S, axis=0)
        
        logits = []
        # Step 2: Generate combinations (S choose S/2)
        combinations = list(itertools.combinations(range(self.S), self.S // 2))
        
        for c in combinations:
            # Form Training Set (J) and Testing Set (J_bar)
            J = np.concatenate([submatrices[i] for i in c], axis=0)
            J_bar = np.concatenate([submatrices[i] for i in range(self.S) if i not in c], axis=0)
            
            # Step 3: In-Sample Optimization (Calculate Sharpe Ratio equivalent for J)
            # We use mean/std of returns as a simple Sharpe proxy
            train_returns = np.mean(J, axis=0) / (np.std(J, axis=0) + 1e-8)
            optimal_idx = np.argmax(train_returns)
            
            # Step 4: Out-of-Sample Evaluation
            test_returns = np.mean(J_bar, axis=0) / (np.std(J_bar, axis=0) + 1e-8)
            
            # Calculate Relative Rank (omega_bar)
            ranks = rankdata(test_returns)
            omega_bar = ranks[optimal_idx] / (self.N + 1) # Normalized rank (0, 1)
            
            # Calculate Logit
            logit = np.log(omega_bar / (1.0 - omega_bar))
            logits.append(logit)
            
        # Final PBO: Ratio of logits less than zero
        logits = np.array(logits)
        pbo = np.mean(logits < 0)
        
        return pbo, logits
