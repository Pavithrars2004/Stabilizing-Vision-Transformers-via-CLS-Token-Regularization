import copy
import torch


class EMA:
    """
    Exponential Moving Average of model parameters.
    Supports state_dict() and load_state_dict() for checkpoint saving.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()

        # EMA model should not require gradients
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA parameters using the current model's parameters.
        """
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())

        for name in ema_params.keys():
            ema_params[name].mul_(self.decay).add_(model_params[name] * (1.0 - self.decay))

    def clone_model(self, model):
        """
        Returns a fresh copy of the EMA model on the same device.
        """
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(self.ema_model.state_dict())
        return new_model

    # ---------------------------------------------------------
    #  IMPORTANT: enable saving/loading
    # ---------------------------------------------------------
    def state_dict(self):
        return {
            "decay": self.decay,
            "ema_model_state": self.ema_model.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.ema_model.load_state_dict(state_dict["ema_model_state"])
