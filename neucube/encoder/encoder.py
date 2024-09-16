import torch
from abc import ABC, abstractmethod

class Encoder(ABC):
    def __init__(self):
        super().__init__()

    def encode_dataset(self, dataset):
        """
        Encodes a dataset using the implemented encoding technique.

        Args:
            dataset (torch.Tensor): Input dataset to be encoded.

        Returns:
            torch.Tensor: Encoded dataset with spike patterns.
        """
        encoded_data = torch.zeros_like(dataset)

        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[2]):
                encoded_data[i][:, j] = self.encode(dataset[i][:, j], feature_index=j)

        return encoded_data

    @abstractmethod
    def encode(self, X_in, **kwargs):
        """
        Encodes an input sample using the implemented encoding technique.

        Args:
            X_in (torch.Tensor): Input sample to be encoded.

        Returns:
            torch.Tensor: Encoded spike pattern for the input sample.
        """
        pass

class RateEncoder(Encoder):
    def __init__(self, min_values, max_values, max_rate=0.5, window_size=10, scale_factor=1.5):
        self.min_values = min_values
        self.max_values = max_values
        self.max_rate = max_rate
        self.window_size = window_size
        self.scale_factor = scale_factor
    def moving_average(self, data):
        if self.window_size < 2:
            return data
        data = data.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, sequence_length]
        padding = self.window_size // 2
        smoothed_data = torch.nn.functional.avg_pool1d(
            data,
            kernel_size=self.window_size,
            stride=1,
            padding=padding,
            count_include_pad=False
        )
        smoothed_data = smoothed_data.squeeze(0).squeeze(0)
        return smoothed_data[:data.shape[-1]]

    def encode(self, sample, feature_index):
        """
        Encodes an input sample using rate encoding based on the negative of the moving average derivative.

        Args:
            sample (torch.Tensor): Input sample to be encoded, shape (time_steps,)
            feature_index (int): Index of the feature for min/max normalization.

        Returns:
            torch.Tensor: Encoded spike train for the input sample, shape (time_steps,)
        """
        # Compute the derivative with the same size as sample
        derivative = torch.zeros_like(sample)
        derivative[1:] = sample[1:] - sample[:-1]
        derivative[0] = 0  # Handle as appropriate for your data

        # Apply moving average to the derivatives
        derivatives = self.moving_average(derivative)

        # Take the negative of the moving average derivative
        negative_derivatives = -derivatives

        # Normalize the negative derivatives using dataset-wide min and max values
        min_val = self.min_values[feature_index]
        max_val = self.max_values[feature_index]
        range_val = max_val - min_val + 1e-10  # Avoid division by zero
        normalized_derivatives = (negative_derivatives - min_val) / range_val
        normalized_derivatives = torch.clamp(normalized_derivatives, 0, 1)

        # Scale the spike probabilities based on the maximum rate
        spike_probs = normalized_derivatives * self.max_rate
        #spike_probs = torch.exp(normalized_derivatives * self.scale_factor) * self.max_rate


        # Generate spikes based on probabilities
        spikes = torch.rand_like(spike_probs) < spike_probs

        return spikes.float()


        
class Delta(Encoder):
  def __init__(self, threshold=0.1):
    """
    Initializes the Delta encoder with a threshold value.

    Args:
      threshold (float, optional): Threshold value for spike generation. Defaults to 0.1.
    """
    self.threshold = threshold

  def encode(self, sample):
    """
    Encodes an input sample using delta encoding.

    Delta encoding compares each element in the sample with its previous element,
    and if the difference exceeds the threshold, it generates a spike (1); otherwise, no spike (0).

    Args:
      sample (torch.Tensor): Input sample to be encoded.

    Returns:
      torch.Tensor: Encoded spike train for the input sample.
    """
    aux = torch.cat((sample[0].unsqueeze(0), sample))[:-1]
    spikes = torch.ones_like(sample) * (sample - aux >= self.threshold)
    return spikes


class Deltav2(Encoder):
    def __init__(self, thresholds, window_size=10):
        """
        Initializes the Delta encoder with thresholds for each feature.

        Args:
            thresholds (list or torch.Tensor): Threshold values for each feature.
            window_size (int, optional): Window size for moving average. Defaults to 10.
        """
        self.thresholds = thresholds
        self.window_size = window_size

    def moving_average(self, data):
        if self.window_size < 2:
            return data
        data = data.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, sequence_length]
        padding = self.window_size // 2
        smoothed_data = torch.nn.functional.avg_pool1d(
            data,
            kernel_size=self.window_size,
            stride=1,
            padding=padding,
            count_include_pad=False
        )
        smoothed_data = smoothed_data.squeeze(0).squeeze(0)
        return smoothed_data[:data.shape[-1]]

    def encode(self, sample, feature_index):
        """
        Encodes an input sample using delta encoding based on the negative moving average derivative.

        Args:
            sample (torch.Tensor): Input sample to be encoded, shape (time_steps,)
            feature_index (int): Index of the feature for thresholding.

        Returns:
            torch.Tensor: Encoded spike train for the input sample, shape (time_steps,)
        """
        # Compute the derivative
        derivative = torch.zeros_like(sample)
        derivative[1:] = sample[1:] - sample[:-1]
        derivative[0] = derivative[1]  # Optionally handle the first derivative

        # Apply moving average to the derivatives
        derivatives = self.moving_average(derivative)

        # Take the negative of the moving average derivative
        negative_derivatives = -derivatives

        # Get the threshold for the current feature
        threshold = self.thresholds[feature_index]

        # Generate spikes where negative derivative exceeds threshold
        spikes = (negative_derivatives >= threshold).float()

        return spikes
