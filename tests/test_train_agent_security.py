import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Mock out dependencies to allow importing train_agent in the sandbox
sys.modules['gymnasium'] = MagicMock()
sys.modules['gymnasium.spaces'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['stable_baselines3.common.env_util'] = MagicMock()
sys.modules['stable_baselines3.common.vec_env'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.decomposition'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()

import train_agent

class TestTrainAgentSecurity(unittest.TestCase):
    def test_validate_path_valid(self):
        """Test that paths within the current directory are accepted."""
        # Current directory relative path
        valid_path = "data/train"
        self.assertEqual(train_agent.validate_path(valid_path, "--data_dir"), valid_path)

        # Current directory absolute path
        valid_abs_path = os.path.abspath("data/train")
        self.assertEqual(train_agent.validate_path(valid_abs_path, "--data_dir"), valid_abs_path)

        # Current directory relative path with ./
        valid_dot_path = "./data/train"
        self.assertEqual(train_agent.validate_path(valid_dot_path, "--data_dir"), valid_dot_path)

    def test_validate_path_invalid(self):
        """Test that paths outside the current directory raise ValueError."""
        # Parent directory
        with self.assertRaises(ValueError) as context:
            train_agent.validate_path("../", "--data_dir")
        self.assertIn("traverses outside the base directory", str(context.exception))

        # Absolute path outside
        outside_path = "/tmp/data"
        with self.assertRaises(ValueError):
            train_agent.validate_path(outside_path, "--data_dir")

        # Traversal with multiple ..
        with self.assertRaises(ValueError):
            train_agent.validate_path("../../etc/passwd", "--data_dir")

    def test_validate_path_none(self):
        """Test that None returns None."""
        self.assertIsNone(train_agent.validate_path(None, "--save_path"))

    @patch('train_agent.parse_args')
    @patch('train_agent.PPO')
    @patch('train_agent.DQN')
    @patch('train_agent.make_vec_env')
    @patch('train_agent.TradingEnv')
    @patch('train_agent.multiprocessing.cpu_count', return_value=1)
    def test_main_with_valid_paths(self, mock_cpu, mock_env, mock_vec, mock_dqn, mock_ppo, mock_parse_args):
        """Test that main runs successfully with valid paths."""
        mock_args = MagicMock()
        mock_args.model = "ppo"
        mock_args.timesteps = 10
        mock_args.data_dir = "data/train"
        mock_args.save_path = "models/my_model"
        mock_parse_args.return_value = mock_args

        # Mock model's learn and save to do nothing
        mock_model_instance = MagicMock()
        mock_ppo.return_value = mock_model_instance

        try:
            train_agent.main()
        except ValueError:
            self.fail("main() raised ValueError unexpectedly!")

    @patch('train_agent.parse_args')
    def test_main_with_invalid_data_dir(self, mock_parse_args):
        """Test that main raises ValueError with invalid data_dir."""
        mock_args = MagicMock()
        mock_args.model = "ppo"
        mock_args.data_dir = "../data/train"
        mock_args.save_path = None
        mock_parse_args.return_value = mock_args

        with self.assertRaises(ValueError):
            train_agent.main()

if __name__ == '__main__':
    unittest.main()
