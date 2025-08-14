from torch.utils.data import Dataset
import torch
import os
import librosa
import numpy as np

class GenderDataset(Dataset):
    def __init__(self, main_dir, sample_rate=16_000):
        """
        Initializes the GenderDataset.

        Args:
        main_dir (str): Path to the main directory containing 'female' and 'male' subdirectories.
        sample_rate (int): The target sample rate for loading audio files. Default is 16 kHz.
        """
        self.main_dir = main_dir
        self.sample_rate = sample_rate

        # Splitting the recordings into female and male categories
        self.female, self.male = self.split_to_arr()

        # Combining all recordings into a single list
        self.all_rec = self.female + self.male

    def split_to_arr(self):
        """
        Lists the audio files in 'female' and 'male' subdirectories.

        Returns:
        tuple: Two lists, one for female recordings and one for male recordings.
        """
        female_recordings = os.listdir(os.path.join(self.main_dir, "female"))  # Female audio files
        male_recordings = os.listdir(os.path.join(self.main_dir, "male"))  # Male audio files
        return female_recordings, male_recordings

    def __len__(self):
        """
        Returns the total number of audio files in the dataset.

        Returns:
        int: Total count of female and male recordings.
        """
        return len(self.female) + len(self.male)

    def __getitem__(self, idx):
        """
        Fetches the MFCC feature tensor and one-hot encoded label for a given index.

        Args:
        idx (int): Index of the audio file to retrieve.

        Returns:
        tuple: A tensor of MFCC features and a one-hot encoded tensor for the label.
        """
        # Determine if the index corresponds to a female or male recording
        male_or_female = "female" if idx < len(self.female) else "male"

        # Get the file name for the corresponding index
        file = self.all_rec[idx]

        # Load the audio file and resample it to the specified sample rate
        audio, sr = librosa.load(os.path.join(self.main_dir, male_or_female, file), sr=self.sample_rate)

        # Extract MFCC features (20 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        # Define the maximum number of frames for 2 seconds of audio
        max_frames = 62  # Computed as 2 * 16000 / 512 (frame size of 512)
        if mfcc.shape[1] > max_frames:
            # Truncate to the maximum frame count if it exceeds the limit
            mfcc = mfcc[:, :max_frames]
        else:
            # Pad with zeros if the frame count is less than the maximum
            padding = max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

        # Convert MFCC features to a PyTorch tensor and transpose it
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).T

        # Assign a label: 0 for female, 1 for male
        label = 0 if male_or_female == "female" else 1

        # Create a one-hot encoded label tensor
        label_tensor = torch.zeros(2)
        label_tensor[label] = 1

        # Return the MFCC tensor and label tensor
        return mfcc_tensor, label_tensor