# algo-trading-bot

## Training

Use `train.py` to train your agents.

### Train PPO Agent
```bash
python train.py --model ppo --timesteps 50000
```

### Train DQN Agent
```bash
python train.py --model dqn --timesteps 50000
```

### Options
- `--model`: `ppo` or `dqn` (Required)
- `--timesteps`: Total training timesteps (Default: 50000)
- `--data_dir`: Directory containing training data (Default: `data/train/`)
- `--save_path`: Path to save the trained model (Default: `models/{model}_trading_bot`)
