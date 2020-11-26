import os
import re
from pathlib import Path
from typing import Union, Dict
import torch


def extract_weights_from_model(
    model: torch.nn.Module,
    from_layer: int,
    path: str,
    prefix_keys_with: str = "module_list.",
):
    """
    Extracts weights from a pretrained model, given the layer from which to extract from, and 
    saves the weights.

    Args:
        model (torch.nn.Module): The pretrained model to extract from.
        from_layer (int): The layer to extract from.
        path (str): The path to save the weights.
        prefix_keys_with (str, optional): prefix to add to the weight keys. Defaults to "module_list.".
    """
    modules = list(model.children())[0][from_layer:]
    if prefix_keys_with is not None:
        data = {}
        for key, value in modules.state_dict().items():
            data[f"{prefix_keys_with}{key}"] = value
    else:
        data = modules.state_dict().items()
    torch.save({"model": data}, path)


def extract_weights_from_checkpoint(
    inp: Union[str, Dict], dest: str, regex: str, exclude_matched: bool = True
):
    """
    Extracts keys (eg: weights, etc) from a torch checkpoint using a regex and saves the result.

    Args:
        inp (Union[str, Dict]): The input. Should be a path to the checkpoint file or the checkpoint data itself.
        dest (str): The path to save the result to.
        regex (str): The regex used to filter the keys.
        exclude_matched (bool, optional): Whether the regex should be used to match items to remove \
            (happens when True) or items to include (happens when False). Defaults to True.
    """
    if type(inp) == str:
        inp = torch.load(inp)

    data = {k: v for k, v in inp.items() if exclude_matched ^ bool(re.match(regex, k))}
    torch.save(data, dest)


def parse_data_cfg(path):
    # Parses the data configuration file
    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        val = val.strip()

        if val.startswith('./'):
            val = val.replace("./", str(Path(path).parent) + os.sep)
        elif val.startswith('../'):
            val = val.replace("../", str(Path(os.path.abspath(path)).parent.parent) + os.sep)

        options[key.strip()] = val

    return options