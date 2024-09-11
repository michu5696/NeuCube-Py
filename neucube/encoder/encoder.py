import torch

class RateEncoder:
    def __init__(self, min_values, max_values, max_rate=0.5, window_size=10):
        self.min_values = min_values
        self.max_values = max_values
        self.max_rate = max_rate
        self.window_size = window_size

    def moving_average(self, data):
        if self.window_size < 2:
            return data
        # Ensure data is in float format for convolution operation
        data = data.float()  # Ensure the input data is of type float
    
        # Reshape data to (batch_size * channels, 1, sequence_length)
        batch_size, seq_len, num_channels = data.shape
        reshaped_data = data.permute(0, 2, 1).reshape(batch_size * num_channels, 1, seq_len)
    
        # Define the convolution weights (moving average filter)
        weights = torch.ones(self.window_size).float() / self.window_size
    
        # Apply conv1d to each channel individually
        smoothed_data = torch.nn.functional.conv1d(reshaped_data, weights.unsqueeze(0).unsqueeze(0), padding=self.window_size // 2)
    
        # Get the new sequence length after convolution
        new_seq_len = smoothed_data.shape[-1]
    
        # Reshape back to original format (batch_size, new_seq_len, num_channels)
        smoothed_data = smoothed_data.view(batch_size, num_channels, new_seq_len).permute(0, 2, 1)

        return smoothed_data

    def encode(self, dataset):
        """
        Encodes an entire dataset using rate encoding based on precomputed min and max values.

        Args:
          dataset (torch.Tensor): Input dataset to be encoded.

        Returns:
          tuple: Two tensors representing spike trains for positive and negative derivatives.
        """
        # Calculate the derivative of the dataset along the time dimension
        derivatives = torch.diff(dataset, dim=1)
        # Apply moving average to the derivatives
        derivatives = self.moving_average(derivatives)

        # Remove padding and handle the derivative size carefully (no padding needed)
        # derivatives = torch.cat([derivatives, torch.zeros((dataset.shape[0], 1, dataset.shape[2]), dtype=dataset.dtype)], dim=1)

        # Separate positive and negative derivatives
        #positive_derivatives = torch.relu(derivatives)  # Keep only positive values, set others to 0
        negative_derivatives = torch.relu(-derivatives)  # Convert negative values to positive (absolute) for encoding

        # Channel-wise normalization
        # Normalize the positive and negative smoothed derivatives using precomputed global min and max
        #normalized_positive = (positive_derivatives - positive_derivatives.min(dim=1, keepdim=True)[0]) / \
        #                      (positive_derivatives.max(dim=1, keepdim=True)[0] - positive_derivatives.min(dim=1, keepdim=True)[0] + 1e-10)
        normalized_negative = (negative_derivatives - negative_derivatives.min(dim=1, keepdim=True)[0]) / \
                              (negative_derivatives.max(dim=1, keepdim=True)[0] - negative_derivatives.min(dim=1, keepdim=True)[0] + 1e-10)

        # Scale the spike probabilities based on the maximum rate
        #positive_spike_probs = normalized_positive * self.max_rate
        negative_spike_probs = normalized_negative * self.max_rate

        # Generate spikes based on probabilities
        #positive_spikes = torch.rand_like(positive_derivatives) < positive_spike_probs
        negative_spikes = torch.rand_like(negative_derivatives) < negative_spike_probs

        #return positive_spikes.float(), 
        return negative_spikes.float()
