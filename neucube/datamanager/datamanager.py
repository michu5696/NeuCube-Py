import numpy as np
import h5py
import pandas as pd
import scipy.interpolate as interp
from scipy.signal import find_peaks
import os
import torch


def resample_data(times, data, target_rate):
    # Resample the data to a target sampling rate using linear interpolation
    desired_time_step = 1 / target_rate
    resampled_times = np.arange(times[0], times[-1], desired_time_step)
    interpolator = interp.interp1d(times, data, axis=1, kind='linear')
    resampled_data = interpolator(resampled_times)
    return resampled_times, resampled_data


def calculate_moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')



def magnitude_to_class(magnitude):
    if magnitude <= 0.2:
        return 0
    #elif magnitude <= 0.2:
    #    return 1
    #elif magnitude <= 0.3:
    #    return 2
    else:
        return 1


class DataManager:
    def __init__(self, params):
        self.source_data_path = params['source_data_path']
        self.samples_path = params['samples_path']
        self.sampling_rate = params['sampling_rate']
        self.batch_duration = params['batch_duration']
        self.data = {}  # Dictionary to hold data for both systems
        self.times = {}  # Dictionary to hold times for both systems
        self.batches = {}  # List to hold data batches
        self.events = {}  # Dictionary to hold detected events for each PZ channel
        self.gt_classes = {}

    def fetch_data(self, chunk_id):
        # Load data for both system 1 and 2
        for system_id in [1, 2]:
            file_path = f'{self.source_data_path}/QS04_023_data{chunk_id}_Elsys{system_id}.mat'
            if not os.path.exists(file_path):
                print(f"File {file_path} not found")
                return False

            try:
                with h5py.File(file_path, 'r') as f:
                    times = f['S/t'][0, :]
                    data = f['S/data']
                    resampled_times, resampled_data = resample_data(times, data, self.sampling_rate)
                    self.times[system_id] = [1000 * t for t in resampled_times]
                    self.data[system_id] = resampled_data
            except Exception as e:
                print(f"Error while loading file {file_path}: {e}")
                return False
        return True

    def split_into_batches(self):
        # Iterate over each system to split data into batches
        for system_id, system_data in self.data.items():
            system_times = self.times[system_id]
            total_chunk_duration = system_times[-1] - system_times[0]
            n_time_points = len(system_times)
            n_batches = total_chunk_duration / self.batch_duration
            points_per_batch = int(n_time_points / n_batches)  # Data points per batch

            # Initialize an empty list for the batches of the current system
            self.batches[system_id] = []

            for start_idx in range(0, n_time_points, points_per_batch):
                end_idx = min(start_idx + points_per_batch, n_time_points)  # Ensure we don't go beyond the array
                time_batch = system_times[start_idx:end_idx]
                data_batch = system_data[:, start_idx:end_idx]

                # Check if the batches are filled with data
                if len(time_batch) == 0 or data_batch.size == 0:
                    print(f"Warning: Empty batch encountered for system {system_id} at index range ({start_idx}, "
                          f"{end_idx})")
                    continue  # Skip empty batches

                # Check the dimensions of the data batch
                expected_shape = (system_data.shape[0], end_idx - start_idx)
                if data_batch.shape != expected_shape:
                    print(f"Warning: Mismatched batch shape for system {system_id} at index range ({start_idx}, "
                          f"{end_idx})")
                    continue  # Skip batches with unexpected shapes

                # Append the time and data batches to the list for the current system
                self.batches[system_id].append((time_batch, data_batch))

            # Check if any batches were created for the current system
            if not self.batches[system_id]:
                print(f"Warning: No valid batches created for system {system_id}")


    def identify_events(self, height_threshold, distance_between_events):
        # Channels to consider for system 2
        channels_to_consider = range(12, 16)

        for batch_num in range(len(self.batches[2])):  # Iterate over all batches
            # Process the specified channels of system 2
            batch = self.batches[2][batch_num]  # Get the specified batch for system 2
            self.events[batch_num] = {}
            for channel in channels_to_consider:
                signal = batch[1][channel, :]  # batch[1] because batch is a tuple (time_batch, data_batch)

                # Find positive and negative peaks
                pos_peaks, _ = find_peaks(signal, height=height_threshold, distance=distance_between_events)
                neg_peaks, _ = find_peaks(-signal, height=height_threshold, distance=distance_between_events)

                # Combine and sort peak indices
                all_peaks = np.sort(np.concatenate((pos_peaks, neg_peaks)))

                events_for_channel = []
                i = 0
                while i < len(all_peaks) - 1:
                    # Check if the next peak is within half a second
                    if abs(batch[0][all_peaks[i + 1]] - batch[0][all_peaks[i]]) <= 500:  # 500 ms
                        # Compute the difference between the two peaks and store it as the magnitude of the event
                        magnitude = abs(signal[all_peaks[i]] - signal[all_peaks[i + 1]])
                        # Store the time of the midpoint between the two peaks as the time of the event
                        time = (batch[0][all_peaks[i]] + batch[0][
                            all_peaks[i + 1]]) / 2  # batch[0] because batch is a tuple (time_batch, data_batch)
                        i += 2  # Skip the next peak as it has been processed
                    else:
                        # Only one peak within half a second, so store its magnitude and time
                        magnitude = abs(signal[all_peaks[i]])
                        time = batch[0][all_peaks[i]]
                        i += 1  # Move to the next peak

                    events_for_channel.append((magnitude, time))

                # Store the identified events for each channel in the current batch
                self.events[batch_num][channel] = events_for_channel

    def generate_gt_classes(self):
        channels_to_consider = range(12, 16)  # Locations (x4)
        for batch_index in range(len(self.batches[2])):
            # Default to an empty events list for each channel
            next_batch = self.events.get(batch_index + 1, {channel: [] for channel in channels_to_consider})
            batch_gt = []
            for channel in channels_to_consider:
                # Get the events at the given location in the next batch
                events = next_batch[channel]
                if len(events) > 0:
                    # Find the maximum magnitude event at this location
                    max_magnitude = max(t[0] for t in events)
                    # Convert the maximum magnitude to a class
                    magnitude_class = magnitude_to_class(max_magnitude)
                else:
                    # If no event at this location, set the class to 0
                    magnitude_class = 0

                batch_gt.append(magnitude_class)

            self.gt_classes[batch_index] = batch_gt


    def process_data(self):
        all_labels = []  # List to store all labels across batches and chunks
        batch_counter = 1  # To track batches for sample file naming
        subsampling_rate = 100  # Adjust this value based on your requirements
    
        # Start from chunk_id 2 as per the requirement
        for chunk_id in range(2, 12):  # Assuming chunks 2 to 11
            if not self.fetch_data(chunk_id):
                print(f"Failed to load data for chunk {chunk_id}. Skipping.")
                continue
    
            self.split_into_batches()
            self.identify_events(height_threshold=0.001, distance_between_events=1000)  # Example values
            self.generate_gt_classes()
    
            # Process each batch for both System 1 (9 channels) and System 2 (16 channels)
            num_channels_system1 = 9
            num_channels_system2 = 16
    
            for batch_index in range(len(self.batches[1]) - 1):  # Assuming system 1 and system 2 have the same number of batches
                # Get data for both systems
                _, data_batch_system1 = self.batches[1][batch_index]
                _, data_batch_system2 = self.batches[2][batch_index]
    
                # Subsample the data for both systems
                subsampled_data_batch_system1 = data_batch_system1[:num_channels_system1, ::subsampling_rate]  # System 1
                subsampled_data_batch_system2 = data_batch_system2[:num_channels_system2, ::subsampling_rate]  # System 2
    
                # Convert the subsampled data from NumPy to PyTorch tensors
                subsampled_data_batch_system1_tensor = torch.tensor(subsampled_data_batch_system1)
                subsampled_data_batch_system2_tensor = torch.tensor(subsampled_data_batch_system2)
    
                # Concatenate the subsampled data from both systems (vertically stack them)
                combined_data_batch = torch.cat((subsampled_data_batch_system1_tensor, subsampled_data_batch_system2_tensor), dim=0)
    
                # Transpose the data to swap dimensions: (channels, time) -> (time, channels)
                combined_data_batch = combined_data_batch.transpose(0, 1)
    
                # Create a DataFrame and save the combined batch to a CSV file
                df_sample = pd.DataFrame(combined_data_batch.numpy(), 
                                         columns=[f'Channel_{i+1}' for i in range(combined_data_batch.shape[1])])
                sample_csv_path = os.path.join(self.samples_path, f'sample_{batch_counter}.csv')
                df_sample.to_csv(sample_csv_path, index=False)
                print(f"Batch {batch_counter} from both systems saved to {sample_csv_path}")
                batch_counter += 1
    
                # Collect labels for each batch (if you have labels for both systems, update this accordingly)
                labels = self.gt_classes.get(batch_index, [0, 0, 0, 0])  # Default to [0, 0, 0, 0] if no data
                all_labels.append(labels)
                
        # Save all labels to a single CSV file
        df_labels = pd.DataFrame(all_labels, columns=['Zone1', 'Zone2', 'Zone3', 'Zone4'])
        labels_csv_path = os.path.join(self.samples_path, 'all_class_labels.csv')
        df_labels.to_csv(labels_csv_path, index=False)
        print(f"All ground truth labels saved to {labels_csv_path}")



