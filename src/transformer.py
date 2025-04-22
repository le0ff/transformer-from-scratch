from typing import Dict, Optional

import numpy as np
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

    @classmethod
    def build_transformer(
        cls,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        dropout_rate: float,
        d_ff: int,
        seed: Optional[int],
    ) -> "Transformer":
        """
        Build a transformer model with the given parameters.
        Parameters:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            src_seq_len (int): Source sequence length.
            tgt_seq_len (int): Target sequence length.
            d_model (int): Dimension of the model.
            n_blocks (int): Number of blocks in the encoder and decoder.
            n_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward network.
        Returns:
            Transformer: A transformer model.
        """
        # Seed for reproducibility
        main_rng = np.random.default_rng(seed)
        max_seed_val = 2**31 - 1
        seeds = main_rng.integers(0, max_seed_val, size=7)

        # Input Embedding
        src_embedding = InputEmbedding(
            d_model=d_model, vocab_size=src_vocab_size, seed=int(seeds[0])
        )
        tgt_embedding = InputEmbedding(
            d_model=d_model, vocab_size=tgt_vocab_size, seed=int(seeds[1])
        )
        # Positional Encoding
        src_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=src_seq_len,
            dropout_rate=dropout_rate,
            seed=int(seeds[2]),
        )
        tgt_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=tgt_seq_len,
            dropout_rate=dropout_rate,
            seed=int(seeds[3]),
        )
        # Encoder Blocks
        encoder = Encoder.from_config(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout_rate,
            num_blocks=n_blocks,
            seed=int(seeds[4]),
        )
        # Decoder Blocks
        decoder = Decoder.from_config(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout_rate,
            num_blocks=n_blocks,
            seed=int(seeds[5]),
        )
        # Projection Layer
        projection = ProjectionLayer(
            d_model=d_model, vocab_size=tgt_vocab_size, seed=int(seeds[6])
        )
        # Transformer Model
        transformer = cls(
            encoder=encoder,
            decoder=decoder,
            src_embed=src_embedding,
            tgt_embed=tgt_embedding,
            src_pos=src_pos_encoding,
            tgt_pos=tgt_pos_encoding,
            projection_layer=projection,
        )
        return transformer

    def encode(self, src: ndarray, src_mask: ndarray) -> ndarray:
        """
        Encode source sequences.

        Applies source embedding, positional encoding, and encoder.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, mask=src_mask)

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
        return self.decoder(
            tgt, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask
        )

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
