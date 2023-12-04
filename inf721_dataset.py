import os
import torchaudio
import torch.nn.functional as F
import torch
import torchvision.transforms as T
import librosa
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

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
        y, sr = librosa.load(audio_path, duration=30)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        if chroma.shape[1] > max_time_dimension:
            chroma = chroma[:, :max_time_dimension]
        else:
            chroma = F.pad(chroma, (0, max_time_dimension - chroma.shape[1]))
        y, sr = librosa.load(audio_path, duration=30)
        return chroma, int(cluster)


audio_folder = './dataset/'
audio_dataset = AudioDataset(audio_folder, target_sample_rate=8000)
print(len(audio_dataset))
chroma, cluster = audio_dataset[871]

print("Cluster:", cluster)
print("chroma shape:", chroma.shape)

def get_data_loaders(audio_folder='./dataset/', batch_size=32):
    audio_dataset = AudioDataset(audio_folder, target_sample_rate=8000)

    y = [int(cluster) for _, cluster in audio_dataset]

    train_indices, testval_indices = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)
    test_indices, val_indices = train_test_split(testval_indices, test_size=0.5, stratify=[y[i] for i in testval_indices], random_state=42)

    train_dataset = Subset(AudioDataset(audio_folder), train_indices)
    test_dataset = Subset(AudioDataset(audio_folder), test_indices)
    val_dataset = Subset(AudioDataset(audio_folder), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

train_loader, test_loader, val_loader = get_data_loaders()

torch.manual_seed(1)

train_batch, train_label_batch = next(iter(train_loader))
test_batch, test_label_batch = next(iter(test_loader))

print("A single training batch size: ", train_batch.size())
print("A single test batch size: ", test_batch.size())
print("Sanity check: ", train_batch[0][0][:5])

