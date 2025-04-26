# Transformer Architecture

This repository implements the Transformer architecture as described in the paper "Attention Is All You Need". The Transformer is a sequence-to-sequence model that relies entirely on self-attention mechanisms, eliminating the need for recurrent or convolutional layers.

## Architecture Overview

The Transformer model processes sequences using the following components:

### 1. Input Processing
- **InputEmbeddings**: Converts token indices to dense vectors and scales them by √d_model for proper scaling
- **PositionalEncoding**: Adds position information using sine/cosine functions, enabling the model to understand token order

### 2. Encoder Path
- **Encoder**: Contains multiple EncoderBlocks that gradually transform the input
- **EncoderBlock**: Consists of two sub-layers with residual connections:
  1. Multi-head self-attention mechanism
  2. Position-wise feed-forward network

### 3. Decoder Path
- **Decoder**: Contains multiple DecoderBlocks
- **DecoderBlock**: Consists of three sub-layers with residual connections:
  1. Masked multi-head self-attention (prevents attending to future tokens)
  2. Multi-head cross-attention, connecting to encoder output
  3. Position-wise feed-forward network

### 4. Output Generation
- **ProjectionLayer**: Linear transformation followed by softmax to predict next token probabilities

## Key Components in Detail

### Multi-Head Attention

The core component of the Transformer architecture:

```python
class MultiHeadAttentionBlock(nn.Module):
    # Implementation details
```

This mechanism:
1. Projects input vectors to query, key, and value vectors using linear transformations
2. Splits these projections into multiple "heads" (default: h=8)
3. Computes scaled dot-product attention for each head independently
4. Concatenates and linearly transforms the results

The attention mechanism calculates:
- Attention scores: `(Q × K^T) / √d_k`
- Applies masking if needed (crucial for decoder self-attention)
- Softmax normalization to get attention weights
- Weighted sum of values: `attention_weights × V`

### Position-wise Feed-Forward Network

```python
class FeedForwardBlock(nn.Module):
    # Implementation details
```

Applied to each position separately and identically:
1. Linear transformation (d_model → d_ff)
2. ReLU activation
3. Dropout
4. Linear transformation (d_ff → d_model)

This functions like a sophisticated 1x1 convolution, allowing the model to learn complex transformations.

### Residual Connections and Layer Normalization

```python
class ResidualConnection(nn.Module):
    # Implementation details
```

For stabilizing training and improving gradient flow:
1. LayerNorm is applied first (pre-norm variant)
2. Then the sublayer (attention or feed-forward)
3. Dropout is applied
4. Finally, residual connection adds the input

### The Build Process

The `build_transformer()` function constructs the entire model:
1. Creates embedding and positional encoding layers
2. Stacks N identical encoder blocks
3. Stacks N identical decoder blocks
4. Creates projection layer
5. Initializes parameters with Xavier uniform initialization

## Implementation Details

1. **Masking**: 
   - Source masks protect padding tokens
   - Target masks prevent looking at future tokens during training

2. **Parameter Sharing**:
   - Layer parameters are independent
   - Embedding layers could be shared (weight tying)

3. **Embedding Scaling**:
   - Embeddings are scaled by √d_model to keep variance stable

4. **Initialization**:
   - Xavier uniform initialization helps with proper gradient flow early in training

## Processing Flow

When handling a sequence pair (e.g., for translation):
1. Source tokens are embedded and positionally encoded
2. This passes through the encoder, creating context-aware representations
3. Target tokens (shifted right) are embedded and positionally encoded
4. The decoder processes these while attending to encoder output
5. The projection layer predicts the next token probabilities

This implementation follows the architecture described in the original paper and can be used for tasks like translation, summarization, or other sequence-to-sequence tasks when properly trained.
