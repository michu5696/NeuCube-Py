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
                encoded_data[i][:, j] = self.encode(dataset[i][:, j])

        return encoded_data

    @abstractmethod
    def encode(self, X_in):
        """
        Encodes an input sample using the implemented encoding technique.

        Args:
          X_in (torch.Tensor): Input sample to be encoded.

        Returns:
          torch.Tensor: Encoded spike pattern for the input sample.
        """
        pass

class RateEncoder(Encoder):
    def __init__(self, max_rate=0.5, window_size=10):
        """
        Initializes the Rate encoder with a maximum spike rate and window size for smoothing.

        Args:
          max_rate (float, optional): Maximum spike rate. Defaults to 0.5.
          window_size (int, optional): Window size for the moving average filter. Defaults to 10.
        """
        self.max_rate = max_rate
        self.window_size = window_size

    def moving_average(self, data):
        if self.window_size < 2:
            return data
        # Ensure data is in float format for convolution operation
        data = data.float()

        # Reshape data to (1, 1, sequence_length) for convolution
        data = data.unsqueeze(0).unsqueeze(0)

        # Define the convolution weights (moving average filter)
        weights = torch.ones(self.window_size).float() / self.window_size

        # Apply conv1d
        smoothed_data = torch.nn.functional.conv1d(
            data, weights.unsqueeze(0).unsqueeze(0), padding=self.window_size // 2
        )

        # Squeeze back to original shape
        smoothed_data = smoothed_data.squeeze(0).squeeze(0)

        return smoothed_data

    def encode(self, sample):
        """
        Encodes an input sample using rate encoding.

        Rate encoding generates spikes based on the rate proportional to the amplitude of the input signal.

        Args:
          sample (torch.Tensor): Input sample to be encoded, shape (time_steps,)

        Returns:
          torch.Tensor: Encoded spike train for the input sample, shape (time_steps,)
        """
        # Calculate the derivative of the sample along the time dimension
        derivatives = torch.diff(sample, dim=0)
        # Apply moving average to the derivatives
        derivatives = self.moving_average(derivatives)

        # Since derivatives is shorter by one than sample due to diff, we can pad it
        derivatives = torch.cat((derivatives[0:1], derivatives))

        # Convert negative derivatives to positive values
        negative_derivatives = torch.relu(-derivatives)

        # Normalize the negative derivatives per sample
        min_val = negative_derivatives.min()
        max_val = negative_derivatives.max()
        normalized_negative = (negative_derivatives - min_val) / (max_val - min_val + 1e-10)

        # Scale the spike probabilities based on the maximum rate
        negative_spike_probs = normalized_negative * self.max_rate

        # Generate spikes based on probabilities
        negative_spikes = torch.rand_like(negative_derivatives) < negative_spike_probs

        return negative_spikes.float()

        
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
