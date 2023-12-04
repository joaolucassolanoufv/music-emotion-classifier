import os
import torchaudio
import torch.nn.functional as F
import torch
import torchvision.transforms as T
import librosa
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")

class AudioDataset(Dataset):
    def __init__(self, audio_folder, target_sample_rate=8000, transform=None):
        self.audio_folder = audio_folder
        self.target_sample_rate = target_sample_rate
        self.transform = transform

        self.file_paths = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith('.mp3')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]

        filename = os.path.basename(audio_path)
        index, cluster = filename.split('-')[:2]
        cluster = cluster.split('.')[0]

        max_time_dimension = 1000
        y, sr = librosa.load(audio_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        if chroma.shape[1] > max_time_dimension:
            chroma = chroma[:, :max_time_dimension]
        else:
            chroma = F.pad(chroma, (0, max_time_dimension - chroma.shape[1]))

        return chroma, int(cluster)
        


audio_folder = './dataset/'
audio_dataset = AudioDataset(audio_folder, target_sample_rate=8000)

y = [int(cluster) for _, cluster in audio_dataset]

train_indices, test_indices = train_test_split(range(len(y)), test_size=0.15, stratify=y, random_state=42)

train_dataset = Subset(AudioDataset(audio_folder), train_indices)
test_dataset = Subset(AudioDataset(audio_folder), test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

save_path = './dataloaders.pth'
torch.save({'train_loader': train_loader, 'test_loader': test_loader}, save_path)
