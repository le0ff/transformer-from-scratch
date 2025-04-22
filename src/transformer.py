from typing import Dict

from numpy import ndarray

from src.decoder import Decoder
from src.encoder import Encoder
from src.layers.base import BaseLayer
from src.layers.embeddings.input_embedding import InputEmbedding
from src.layers.embeddings.positional_encoding import PositionalEncoding
from src.layers.projection import ProjectionLayer


class Transformer(BaseLayer):
    """
    Transformer model
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

        self._layers = {
            "encoder": encoder,
            "decoder": decoder,
            "src_embed": src_embed,
            "tgt_embed": tgt_embed,
            "src_pos": src_pos,
            "tgt_pos": tgt_pos,
            "projection_layer": projection_layer,
        }

    def encode(self, src: ndarray, src_mask: ndarray) -> ndarray:
        """
        Encode source sequences.

        Applies source embedding, positional encoding, and encoder.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: ndarray,
        src_mask: ndarray,
        tgt: ndarray,
        tgt_mask: ndarray,
    ) -> ndarray:
        """
        Decode target sequences.

        Applies target embedding, positional encoding, and decoder.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: ndarray) -> ndarray:
        """
        Project the output to the target vocabulary size.

        Applies a linear transformation to the decoder output.
        """
        return self.projection_layer(x)

    def forward(
        self, src: ndarray, tgt: ndarray, src_mask: ndarray, tgt_mask: ndarray
    ) -> ndarray:
        """
        Forward pass through the transformer model.

        Applies encoding, decoding, and projection.

        Parameters:
            src (ndarray): Source sequences.
            tgt (ndarray): Target sequences.
            src_mask (ndarray): Source mask.
            tgt_mask (ndarray): Target mask.

        Returns:
            ndarray: Projected output (probability distribution over target vocabulary).
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)

    def __call__(
        self, src: ndarray, tgt: ndarray, src_mask: ndarray, tgt_mask: ndarray
    ) -> ndarray:
        """
        Call method to enable the use of the model as a function.
        """
        return self.forward(src, tgt, src_mask, tgt_mask)

    def train(self) -> None:
        """Set transformer model to training mode."""
        super().train()
        for layer in self._layers.values():
            layer.train()

    def eval(self) -> None:
        """Set transformer model to evaluation mode."""
        super().eval()
        for layer in self._layers.values():
            layer.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """
        Get all parameters of the transformer model.

        Returns:
            Dict[str, ndarray]: Dictionary of parameters with their names as keys.
        """
        pass

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """
        Set parameters of the transformer model.

        Parameters:
            params (Dict[str, ndarray]): Dictionary of parameters with their names as keys.
        """
        pass
