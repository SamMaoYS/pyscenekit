import os
import torch


def load_ckpt(
    load_path,
    model,
    strict_match=True,
):
    """
    Load the check point for resuming training or finetuning.
    """
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path, map_location="cpu")
        ckpt_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(ckpt_state_dict, strict=strict_match)

        del ckpt_state_dict
        del checkpoint
    else:
        raise FileNotFoundError(f"File not found: {load_path}")
    return model
