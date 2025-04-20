from typing import Dict

from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.normalization import LayerNorm


class Encoder(BaseLayer):
    """
    Encoder for the Transformer model.

    Consists of multiple encoder blocks, each containing a multi-head self-attention layer, feedforward block, residual connections, and dropout.
    """

    def __init__(self, layers: Dict[str, BaseLayer]) -> None:
        """
        Initializes the Encoder.

        Parameters:
            layers (Dict[str, BaseLayer]): Dictionary of layers in the encoder (in sequential order).
        """
        super().__init__()
        # dictionary of encoder blocks, e.g. {"encoder_block_1": EncoderBlock(...), ...}
        self.layers = layers
        # LayerNormalization, using the d_model of the first encoder block
        self.norm = LayerNorm(layers.values()[0].d_model)

    def forward(self, x: ndarray, mask: ndarray) -> ndarray:
        """
        Forward pass through the encoder.

        Parameters:
            x (ndarray): Input data.
            mask (ndarray): Mask for the input data.

        Returns:
            ndarray: Output data after passing through the encoder.
        """
        for layer in self.layers.values():
            x = layer(x, mask)
        return self.norm(x)

    def train(self) -> None:
        """Set the layer to training mode."""
        super().train()
        for layer in self.layers.values():
            layer.train()
        self.norm.train()

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        super().eval()
        for layer in self.layers.values():
            layer.eval()
        self.norm.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """Get all parameters from the encoder, namespaced by block and sublayer."""
        params = {}
        for block_name, block in self.layers.items():
            block_params = block.get_parameters()
            for key, value in block_params.items():
                params[f"{block_name}_{key}"] = value
        # Add LayerNorm parameters for the encoder output norm
        norm_params = self.norm.get_parameters()
        for key, value in norm_params.items():
            params[f"norm_{key}"] = value
        return params

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Set parameters for the encoder, expecting namespaced keys."""
        if not params:
            raise ValueError("No parameters provided for Encoder.")

        # Prepare dicts for each block and for norm
        block_param_dicts = {name: {} for name in self.layers}
        norm_param_dict = {}
        processed_keys = set()

        for key, value in params.items():
            matched = False
            # Check if key belongs to a block
            for block_name in self.layers:
                prefix = f"{block_name}_"
                if key.startswith(prefix):
                    block_param_dicts[block_name][key[len(prefix) :]] = value
                    processed_keys.add(key)
                    matched = True
                    break
            # Check if key belongs to norm
            if not matched and key.startswith("norm_"):
                norm_param_dict[key[len("norm_") :]] = value
                processed_keys.add(key)
                matched = True
            if not matched:
                raise ValueError(f"Unexpected parameter key for Encoder: {key}")

        # Set parameters for each block
        for block_name, block_params in block_param_dicts.items():
            if block_params:
                self.layers[block_name].set_parameters(block_params)
        # Set parameters for norm
        if norm_param_dict:
            self.norm.set_parameters(norm_param_dict)

        # Check for missing keys
        if len(processed_keys) != len(params):
            missing = set(params.keys()) - processed_keys
            raise ValueError(f"Some parameters were not processed: {missing}")
