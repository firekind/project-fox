import torch


def extract_weights_from_model(
    model: torch.nn.Module, from_layer: int, path: str, prefix_keys_with: str ="module_list."
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

