import numpy as np
import torch
import torch.nn.functional as F

def rmse_loss(output, target, letterbox_borders = None):
    if letterbox_borders is not None:
        output = crop_and_resize(output, letterbox_borders, target.shape[1:])
    
    return torch.sqrt(
        F.mse_loss(output, target)
    )


def crop_and_resize(output, letterbox_borders, target_shape):
    new_output = torch.zeros(output.size(0), target_shape[0], target_shape[1], target_shape[2], device=output.device)

    for i, o in enumerate(output):
        dw, dh = letterbox_borders[i]
        if not (dw == 0 and dh == 0):
            if dw == 0:
                dh = int(np.ceil(dh))
                o = o[dh:-dh, :]
            else:
                dw = int(np.ceil(dw))
                o = o[:, dw:-dw]
        new_output[i] = F.interpolate(
            o.unsqueeze(0).unsqueeze(0), # adding channel and batch dimension
            (target_shape[1], target_shape[2])
        ).squeeze(0).squeeze(0) # removing channel and batch dimension

    return new_output