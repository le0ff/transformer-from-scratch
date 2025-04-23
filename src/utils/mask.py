import numpy as np
from numpy import ndarray


def create_src_mask(
    src_tokens, pad_token_id, n_heads, mask_keys_only: bool = True
) -> ndarray:
    seq_len = len(src_tokens)
    # mask for padding tokens
    mask = np.array([1 if t != pad_token_id else 0 for t in src_tokens], dtype=np.uint8)
    if not mask_keys_only:
        # outer product to create a 2D mask for padding tokens
        mask = np.outer(mask, mask)
    # add dimensions for broadcasting
    mask = mask[np.newaxis, np.newaxis, :]
    # broadcast to (batch_size, n_heads, seq_len, seq_len)
    mask = np.broadcast_to(mask, (1, n_heads, seq_len, seq_len))
    return mask


def create_tgt_mask(
    tgt_tokens, pad_token_id, n_heads, mask_keys_only: bool = True
) -> ndarray:
    seq_len = len(tgt_tokens)
    # lower triangular matrix for causal mask
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.uint8))
    # mask for padding tokens
    pad_mask = np.array(
        [1 if t != pad_token_id else 0 for t in tgt_tokens], dtype=np.uint8
    )
    if mask_keys_only:
        pad_mask_matrix = pad_mask[np.newaxis, :]
    else:
        # outer product to create a 2D mask for padding tokens
        pad_mask_matrix = np.outer(pad_mask, pad_mask)

    # combine causal mask and padding mask
    combined_mask = causal_mask * pad_mask_matrix
    # add dimensions for broadcasting
    mask = combined_mask[np.newaxis, np.newaxis, :, :]
    # broadcast to (batch_size, n_heads, seq_len, seq_len)
    mask = np.broadcast_to(mask, (1, n_heads, seq_len, seq_len))
    return mask
