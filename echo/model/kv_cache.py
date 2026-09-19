import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed.
                Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2, start=None):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation
                should be done. Default is 2.
            start (int, optional): Python cache length. Avoids Tensor.item()
                inside torch.compile. If omitted, uses self.current_length.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        q_len = tensor.shape[dim]
        if start is None:
            dst = self.data.narrow(dim, self.current_length, q_len)
            dst.copy_(tensor, non_blocking=True)
            self.current_length.add_(q_len)
            return torch.narrow(self.data, 2, 0, self.current_length)
        dst = self.data.narrow(dim, start, q_len)
        dst.copy_(tensor, non_blocking=True)
        return self.data.narrow(2, 0, start + q_len)
    
    def rollback(self, length: int):
        """
        Reset the current length to a specific value.
        This effectively discards any data after 'length'.
        """
        if isinstance(self.current_length, torch.Tensor):
            self.current_length.fill_(length)
        else:
            self.current_length = length


def initialize_past_key_values(model, max_length=32768):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to
    store and reuse past key and value states during autoregressive decoding,
    which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states
        need to be initialized.

    Returns:
        tuple:
            - past_key_values (list):
                A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor):
                The tensor that will store all keys and values.
            - current_length_data (torch.Tensor):
                 A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different
    # batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    
    if config.max_position_embeddings < max_length:
        max_length = config.max_position_embeddings
        
    if hasattr(model, "layers"):
        decoder_layers = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        decoder_layers = model.language_model.layers
    else:
        raise NotImplementedError(f"Unsupported model architecture: {model}")
    
    devices = [layer.self_attn.q_proj.weight.device for layer in decoder_layers]
    
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]
    for _, i in enumerate(devices):
        if startdevice != i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                max_length,
                head_dim,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum = 0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        head_dim,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers.
    # Walk the same consecutive-device runs used when allocating shards.
    # Do not use device.index: `cuda` (no index) and `cpu` are None, and
    # `cuda` vs `cuda:0` would also split shards while .index arithmetic
    # still treats them as one buffer.
    past_key_values = []
    list_idx = 0
    bias = 0
    prev_device = devices[0]
    for i in range(config.num_hidden_layers):
        if devices[i] != prev_device:
            list_idx += 1
            bias = 0
            prev_device = devices[i]
        shard = past_key_values_data_list[list_idx]
        past_key_values.append(
            [
                KVCache(shard[2 * bias + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
        bias += 1
    return past_key_values, past_key_values_data_list, current_length_data
