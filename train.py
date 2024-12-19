import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import gc
import random
from torchaudio.models.wav2vec2 import wav2vec2_base

class LibriSpeechDataset(Dataset):
    def __init__(self, csv_path, max_length=16000*5, subset_size=None):  # 5 seconds at 16kHz
        self.data = pd.read_csv(csv_path)
        if subset_size is not None:
            if subset_size > len(self.data):
                print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(self.data)}. Using full dataset.")
            else:
                self.data = self.data.sample(n=min(subset_size, len(self.data)), 
                                          random_state=42).reset_index(drop=True)
        self.max_length = max_length
        self.wav2vec_processor = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=16000
        )
        
        print(f"Dataset size: {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def create_realistic_labels(self, num_frames, num_speakers=2):
        """Create more realistic speaker activity patterns"""
        labels = torch.zeros((num_speakers, num_frames))
        
        # Create segments where different speakers are active
        min_segment_length = num_frames // 10
        current_pos = 0
        
        while current_pos < num_frames:
            segment_length = random.randint(min_segment_length, num_frames // 4)
            if current_pos + segment_length > num_frames:
                segment_length = num_frames - current_pos
                
            active_speaker = random.randint(0, num_speakers - 1)
            overlap = random.random() < 0.2
            
            labels[active_speaker, current_pos:current_pos + segment_length] = 1
            if overlap and segment_length > min_segment_length:
                other_speaker = (active_speaker + 1) % num_speakers
                overlap_length = segment_length // 2
                labels[other_speaker, current_pos:current_pos + overlap_length] = 1
                
            current_pos += segment_length
        
        return labels
    
    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            waveform, sample_rate = torchaudio.load(row['processed_path'])
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                waveform = self.wav2vec_processor(waveform)
            
            # Pad or truncate to max_length
            if waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, :self.max_length]
            
            # Calculate number of frames based on Wav2Vec2 downsampling factor
            num_frames = self.max_length // 320  # Wav2Vec2 downsampling factor
            labels = self.create_realistic_labels(num_frames)
            
            return waveform, labels, row['speaker_id']
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a zero tensor of appropriate size as fallback
            return (torch.zeros((1, self.max_length)), 
                   torch.zeros((2, self.max_length // 320)), 
                   -1)

class PretrainedEENDModel(nn.Module):
    def __init__(self, num_speakers=2, freeze_encoder=True):
        super(PretrainedEENDModel, self).__init__()
        # Load pretrained Wav2Vec2
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()
        
        # Freeze encoder layers if specified
        if freeze_encoder:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        
        # New layers for diarization
        self.feature_projection = nn.Linear(768, 256)  # Wav2Vec2 output dim is 768
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(256)
        self.fc = nn.Linear(256, num_speakers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Remove channel dimension and transpose to (batch, time)
        x = x.squeeze(1)  # Remove channel dimension (batch, 1, time) -> (batch, time)
        
        # Calculate lengths for each sequence
        lengths = torch.full((x.shape[0],), x.shape[1], device=x.device)
        
        # Extract features using Wav2Vec2
        with torch.set_grad_enabled(not self.wav2vec.training):
            wav2vec_out, _ = self.wav2vec.extract_features(x, lengths)
        
        # Project features
        x = self.feature_projection(wav2vec_out[-1])  # Use the last layer's output
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Final prediction
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        output = output.transpose(1, 2)
        
        return output
    
    def forward(self, x):
        # Extract features using Wav2Vec2
        with torch.set_grad_enabled(not self.wav2vec.training):
            wav2vec_out = self.wav2vec.extract_features(x)[0]
        
        # Project features
        x = self.feature_projection(wav2vec_out)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Final prediction
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        output = output.transpose(1, 2)
        
        return output

class DiarizationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1):
        super(DiarizationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, predictions, targets):
        # BCE loss
        bce_loss = self.bce(predictions, targets)
        
        # Dynamic class weights
        pos_weight = (targets.sum(dim=2, keepdim=True) + 1e-6) / (targets.shape[2] + 1e-6)
        neg_weight = 1 - pos_weight
        
        # Weighted BCE
        weighted_bce = pos_weight * bce_loss * targets + neg_weight * bce_loss * (1 - targets)
        
        # Overlap penalty
        overlap_penalty = torch.sum(predictions * torch.roll(predictions, 1, dims=1), dim=1).mean()
        
        # Smoothness regularization
        smoothness_penalty = torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1]).mean()
        
        total_loss = weighted_bce.mean() + self.alpha * overlap_penalty + self.beta * smoothness_penalty
        return total_loss

def train_eend(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = DiarizationLoss()
    
    # Different learning rates for pretrained and new layers
    pretrained_params = list(model.wav2vec.parameters())
    new_params = list(model.feature_projection.parameters()) + \
                list(model.lstm.parameters()) + \
                list(model.fc.parameters())
    
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': 1e-5},
        {'params': new_params, 'lr': 1e-4}
    ], weight_decay=0.01)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    max_grad_norm = 1.0
    best_val_loss = float('inf')
    best_model_path = 'best_eend_model.pth'
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, (waveform, labels, _) in enumerate(tqdm(train_loader)):
            try:
                waveform = waveform.to(device)
                labels = labels.to(device)
                
                # Print shapes for debugging
                if batch_idx == 0:
                    print(f"Waveform shape: {waveform.shape}")
                    print(f"Labels shape: {labels.shape}")
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = model(waveform)
                    loss = criterion(output, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                del waveform, labels, output, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for waveform, labels, _ in val_loader:
                try:
                    waveform = waveform.to(device)
                    labels = labels.to(device)
                    
                    output = model(waveform)
                    loss = criterion(output, labels)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    del waveform, labels, output, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    print(f"Error in validation: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Early stopping with patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f'Saved best model with validation loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement')
                break

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset and DataLoader setup
    train_dataset = LibriSpeechDataset('speech_project/data/metadata/train_split.csv', subset_size=200)
    val_dataset = LibriSpeechDataset('speech_project/data/metadata/val_split.csv', subset_size=50)
    
    batch_size = min(16, len(train_dataset) // 10)  # Ensure at least 10 batches
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,  # Reduced num_workers for smaller dataset
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    model = PretrainedEENDModel(num_speakers=2, freeze_encoder=True)
    model = model.to(device)
    print("Model architecture:")
    print(model)
    train_eend(model, train_loader, val_loader, num_epochs=5, device=device)

if __name__ == "__main__":
    main()