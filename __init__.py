import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')


sweep_config = load_config('sweep.yaml')