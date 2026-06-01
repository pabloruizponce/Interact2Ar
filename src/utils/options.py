import argparse

import yaml


def get_options(opt_path: str) -> argparse.Namespace:
    """
    Load configuration from a YAML file.

    Args:
        opt_path (str): Path to the YAML configuration file.

    Returns:
        Namespace: Configuration parameters as a dictionary.
    """
    with open(opt_path, "r") as file:
        opt = yaml.safe_load(file)

    return argparse.Namespace(**opt)
