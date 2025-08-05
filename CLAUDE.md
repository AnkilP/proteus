# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
RL4LMs is a modular reinforcement learning library for fine-tuning language models to human preferences. It provides implementations of on-policy algorithms (PPO, NLPO, A2C, TRPO), reward functions, metrics, and datasets for various NLP tasks.

## Common Development Commands

### Installation
```bash
# Local installation
pip install -e .

# Install additional dependencies for SPICE metric
cd rl4lms/envs/text_generation/caption_metrics/spice && bash get_stanford_models.sh
cd -

# Download required models
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm
```

### Training Models
```bash
# Train PPO/NLPO using YAML configs
python scripts/training/train_text_generation.py --config_path <PATH_TO_CONFIG>

# Example: Train T5-base on CNN/DM summarization with PPO
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/summarization/t5_ppo.yml

# With experiment tracking
WANDB_API_KEY=<YOUR_KEY> python scripts/training/train_text_generation.py \
  --config_path <CONFIG> \
  --experiment_name <NAME> \
  --base_path_to_store_results <PATH> \
  --log_to_wandb
```

### Docker
```bash
# Build Docker image
docker build . -t rl4lms

# Run container
docker run -it rl4lms
```

## Architecture and Key Components

### Core Structure
- **rl4lms/algorithms/**: On-policy RL algorithm implementations (PPO, NLPO, A2C, TRPO)
- **rl4lms/envs/text_generation/**: 
  - `env.py`: Gym-style text generation environment
  - `reward.py`: Base reward function class and implementations
  - `policy.py`: LM-based actor-critic policies (causal and seq2seq)
  - `metric.py`: Evaluation metrics
  - `registry.py`: Component registry for datasets, rewards, metrics
- **rl4lms/data_pools/**: Dataset loaders and task-specific utilities
- **scripts/training/**: 
  - `train_text_generation.py`: Main training entry point
  - `task_configs/`: YAML configurations for different NLP tasks

### Training Flow
1. User provides YAML config specifying task, model, algorithm, and hyperparameters
2. `train_text_generation.py` loads config and instantiates:
   - Tokenizer and data pool
   - RL environment with reward function
   - On-policy algorithm with LM policy
3. `OnPolicyTrainer` or `SupervisedTrainer` manages the training loop:
   - Collects rollouts from parallel environments
   - Computes rewards and updates policy
   - Evaluates on validation set periodically

### Key Design Patterns
- All components (datasets, rewards, metrics, algorithms) are registered in `registry.py`
- Policies wrap HuggingFace transformers models with RL-specific functionality
- Environment processes multiple episodes in parallel using `SubProcVecEnv`
- Adaptive KL controller maintains similarity to original LM
- Supports both causal LMs (GPT-style) and seq2seq models (T5, BART)

### Customization Points
- Custom datasets: Subclass `TextGenPool` and implement `prepare()` method
- Custom rewards: Subclass `RewardFunction` and implement `__call__()`
- Custom metrics: Subclass `BaseMetric` and implement `compute()`
- Custom algorithms: Subclass `OnPolicyAlgorithm` from stable-baselines3
- Register new components in `registry.py` to use in YAML configs

## Available Tasks and Configurations
Pre-configured tasks in `scripts/training/task_configs/`:
- summarization/ (CNN/DM dataset)
- generative_commonsense_reasoning/ (CommonGen)
- imdb_text_continuation/
- table_to_text/ (ToTTo)
- question_answering/ (NarrativeQA)
- machine_translation/ (IWSLT2017)
- dialogue_generation/
- synthetic tasks (generate_increasing_numbers/, generate_dates/)