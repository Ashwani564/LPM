"""
Utility functions for the Mississippi AI Job Impact model.
Initialization helpers registered with the AgentTorch Registry.
"""

import pandas as pd
import torch


def load_population_attribute(shape, params):
    """Load a population attribute from a pickle file."""
    file_path = params["file_path"]
    df = pd.read_pickle(file_path)
    if isinstance(df, pd.Series):
        tensor = torch.from_numpy(df.values).float()
    else:
        tensor = torch.from_numpy(df.values).float()
    return tensor.unsqueeze(1) if tensor.dim() == 1 else tensor
