"""
Classifier head for DINOv2 features and CAM generation.
Provides class-aware evidence to complement anomaly localization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict
from tqdm import tqdm

class DiseaseClassifierHead(nn.Module):
    """
    Linear probe head for DINOv2 features.
    Used to generate Class Activation Maps (CAMs).
    """
    
    def __init__(self, input_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for global features.
        x: (B, D)
        returns: (B, num_classes)
        """
        return self.linear(x)

    def train_head(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        epochs: int = 20, 
        batch_size: int = 64,
        lr: float = 0.001
    ):
        """
        Train the linear head on extracted features.
        features: (N, D)
        labels: (N,) integers [0..num_classes-1]
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        X = torch.from_numpy(features).float().to(self.device)
        y = torch.from_numpy(labels).long().to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"[INFO] Training classifier head on {len(X)} samples for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = self(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += (out.argmax(dim=1) == batch_y).sum().item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}, Acc={correct/len(X):.4f}")
        
        self.eval()
        print("[OK] Classifier head trained successfully")

    def generate_cam(self, patch_features: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Generate CAM for a specific class.
        patch_features: (B, D, H_p, W_p)
        class_idx: Index of the class to generate CAM for
        returns: (B, 1, H_p, W_p) normalized CAM
        """
        self.eval()
        self.to(patch_features.device)
        B, D, H_p, W_p = patch_features.shape
        
        # Reshape to (B, H_p*W_p, D)
        x = patch_features.permute(0, 2, 3, 1).reshape(-1, D)
        
        # Apply linear layer
        with torch.no_grad():
            logits = self.linear(x) # (B*H*W, num_classes)
            
            # Use softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B*H*W, num_classes)
            
            # Select the specified class
            class_prob = probs[:, class_idx] # (B*H*W,)
            
            # Reshape back to (B, 1, H_p, W_p)
            cam = class_prob.reshape(B, 1, H_p, W_p)
            
        return cam

    def save(self, path: Union[str, Path]):
        """Save weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"[INFO] Classifier head saved to {path}")

    def load(self, path: Union[str, Path], device: str = "cpu"):
        """Load weights from disk."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.device = device
        self.to(device)
        self.eval()
        print(f"[INFO] Classifier head loaded from {path}")
