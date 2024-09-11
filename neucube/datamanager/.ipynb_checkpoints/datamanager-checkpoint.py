import numpy as np
import h5py
import pandas as pd
import scipy.interpolate as interp
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import os


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


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, method="pad", padlen=500)
    return y


def rate_encoding(signal, time_batch, min_val, max_val):
    # Normalize the signal
    normalized_signal = (signal - min_val) / (max_val - min_val)
    # Ensure no negative rates by clipping values
    normalized_signal = np.clip(normalized_signal, 0, 1)
    # Calculate the rate
    rate = normalized_signal * 100  # Example rate scaling factor
    spikes = []
    # max_spikes_per_interval = 10  # Set a limit on the number of spikes per interval
    for i in range(1, len(time_batch)):
        # Distribute spikes randomly within the current interval
        # num_spikes = min(int(rate[i]), max_spikes_per_interval)
        num_spikes = int(rate[i])
        if num_spikes > 0:
            spike_times = np.random.uniform(time_batch[i - 1], time_batch[i], num_spikes)
            spikes.extend(spike_times)
    print(len(spikes))
    return sorted(spikes)


def magnitude_to_class(magnitude):
    if magnitude <= 0.2:
        return 0
    elif magnitude <= 0.2:
        return 1
    elif magnitude <= 0.3:
        return 2
    else:
        return 3


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
        self.encoded_spikes = {}
        self.current_min = {}  # To store minimum derivatives for each channel and system_id
        self.current_max = {}  # To store maximum derivatives for each channel and system_id

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

    def process_and_save_batches(self):
        chunk_ids = range(1, 12)

        batch_counter = 1
        for chunk_id in chunk_ids:
            if not self.fetch_data(chunk_id):
                print(f"Failed to load data for chunk {chunk_id}. Skipping.")
                continue

            self.split_into_batches()
            for system_id, batch_list in self.batches.items():
                for batch in batch_list:
                    time_batch, data_batch = batch

                    # Exclude specified channels for system_id 2
                    if system_id == 2:
                        data_batch = np.delete(data_batch, [12, 13, 14, 15],
                                               axis=0)  # Adjusting indices to Python's 0-based indexing

                    # Prepare DataFrame for CSV
                    df = pd.DataFrame(data_batch.transpose(),
                                      columns=[f'Channel_{i + 1}' for i in range(data_batch.shape[0])])
                    df.insert(0, 'Time', time_batch)  # Insert time column at the beginning

                    # Save to CSV
                    csv_path = os.path.join(self.samples_path, f'sample_{batch_counter}.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"Batch {batch_counter} saved to {csv_path}")
                    batch_counter += 1

        print(f"All batches processed and saved. Total batches: {batch_counter - 1}")


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