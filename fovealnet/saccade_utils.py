import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import Counter

class SequenceDataset(Dataset):
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths 
    
    def __len__(self):
        return len(self.folder_paths)
    
    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        images = []
        labels = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".npy"): 
                image_path = os.path.join(folder_path, filename)
                image = np.load(image_path)
                image = torch.tensor(image).unsqueeze(0).float() 
                images.append(image)

                label = 1 if 'SAC' in filename else 0
                labels.append(label)
        if len(images) == 0:
            raise ValueError(f"No .npy files found in {folder_path}")
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels

def collate_fn(batch):
    images, labels = zip(*batch)
    padded_images = pad_sequence(images, batch_first=True, padding_value=0)  
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1) 
    return padded_images, padded_labels

def create_dataloader(root_folder, batch_size=4):
    folder_paths = [os.path.join(root_folder, folder_name) for folder_name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder_name))]
    dataset = SequenceDataset(folder_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn) 
    return dataloader

def compute_class_weights(labels):
    valid_labels = labels[labels >= 0]
    
    label_count = Counter(valid_labels.flatten().cpu().numpy())
    
    total_samples = sum(label_count.values())
    class_weights = {cls: total_samples / count for cls, count in label_count.items()}
    
    weight_tensor = torch.tensor([class_weights.get(i, 0) for i in range(2)]).cuda()  # 只考虑类别0和1
    
    return weight_tensor

def train_model_saccade(model, dataloader, num_epochs, learning_rate = 1e-3):
    all_labels = []
    for images, labels in dataloader:
        all_labels.extend(labels.view(-1).cpu().numpy())
    class_weights = compute_class_weights(torch.tensor(all_labels))

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []

        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for images, labels in tepoch:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs.view(-1, 2), labels.view(-1))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 2) 

                valid_mask = labels.view(-1) != -1
                valid_labels = labels.view(-1)[valid_mask]
                valid_preds = predicted.view(-1)[valid_mask]

                total += valid_labels.size(0)
                correct += (valid_preds == valid_labels).sum().item()

                all_labels.extend(valid_labels.cpu().numpy())
                all_preds.extend(valid_preds.cpu().numpy())

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (len(all_labels) + 1), accuracy=correct / total * 100)

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall)

        epoch_loss = running_loss / len(dataloader)
        accuracy = correct / total * 100
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'True Positives (TP): {tp}, False Positives (FP): {fp}, True Negatives (TN): {tn}, False Negatives (FN): {fn}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, f1: {f1:.4f}')

    print('Training complete')

if __name__ =='__main__':
    dataset = SequenceDataset(['../openeds/train/binary_bit/6400'])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}")
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)