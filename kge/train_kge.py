"""
Training script for Knowledge Graph Embedding models
"""
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kge.models import get_model
except ImportError:
    from models import get_model


class TripleDataset(Dataset):
    """Dataset for KG triples"""
    def __init__(self, triples):
        self.triples = triples
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]


class KGETrainer:
    """Trainer for Knowledge Graph Embedding models"""
    
    def __init__(self, model, n_entities, n_relations, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.best_mrr = 0
        self.patience_counter = 0
    
    def generate_negative_samples(self, batch, n_negative=1):
        """Generate negative samples by corrupting head or tail"""
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        batch_size = len(h)
        
        neg_samples = []
        for _ in range(n_negative):
            # Randomly corrupt head or tail
            corrupt_head = torch.rand(batch_size, device=self.device) < 0.5
            
            neg_h = h.clone()
            neg_t = t.clone()
            
            # Corrupt heads
            n_corrupt_head = corrupt_head.sum().item()
            if n_corrupt_head > 0:
                neg_h[corrupt_head] = torch.randint(
                    0, self.n_entities, 
                    (n_corrupt_head,), 
                    device=self.device
                )
            
            # Corrupt tails
            n_corrupt_tail = (~corrupt_head).sum().item()
            if n_corrupt_tail > 0:
                neg_t[~corrupt_head] = torch.randint(
                    0, self.n_entities, 
                    (n_corrupt_tail,), 
                    device=self.device
                )
            
            neg_samples.append(torch.stack([neg_h, r, neg_t], dim=1))
        
        return torch.cat(neg_samples, dim=0)
    
    def train_epoch(self, train_loader, optimizer, n_negative=1):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Positive samples
            pos_h, pos_r, pos_t = batch[:, 0], batch[:, 1], batch[:, 2]
            pos_score = self.model(pos_h, pos_r, pos_t)
            
            # Negative samples
            neg_batch = self.generate_negative_samples(batch, n_negative)
            neg_h, neg_r, neg_t = neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2]
            neg_score = self.model(neg_h, neg_r, neg_t)
            
            # Loss calculation
            if hasattr(self.model, 'margin'):
                # Margin ranking loss for TransE, TransH, RotatE, CompoundE
                loss = torch.mean(F.relu(self.model.margin + pos_score - neg_score))
            else:
                # Binary cross-entropy for ComplEx
                loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Normalize entity embeddings for TransE
            if isinstance(self.model, torch.nn.Module) and hasattr(self.model, 'entity_emb'):
                if 'TransE' in self.model.__class__.__name__:
                    self.model.entity_emb.weight.data = F.normalize(
                        self.model.entity_emb.weight.data, p=2, dim=1
                    )
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / n_batches
    
    def validate(self, valid_data, batch_size=128):
        """Quick validation using MRR"""
        try:
            from kge.metrics import KGEMetrics
        except ImportError:
            from metrics import KGEMetrics
        
        self.model.eval()
        evaluator = KGEMetrics(self.model, self.device)
        
        # Sample a subset for faster validation
        sample_size = min(len(valid_data), 500)
        indices = np.random.choice(len(valid_data), sample_size, replace=False)
        valid_sample = valid_data[indices]
        
        ranks, _ = evaluator.rank_triples(valid_sample, batch_size=batch_size)
        metrics = evaluator.calculate_metrics(ranks)
        
        return metrics['MRR']
    
    def train(self, train_data, valid_data, n_epochs=100, batch_size=128, 
              lr=0.001, n_negative=1, patience=10, save_dir='results/embeddings'):
        """
        Complete training loop
        
        Args:
            train_data: Training triples
            valid_data: Validation triples
            n_epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            n_negative: Number of negative samples per positive
            patience: Early stopping patience
            save_dir: Directory to save model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loaders
        train_dataset = TripleDataset(train_data)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'valid_mrr': [],
            'best_epoch': 0
        }
        
        print(f"\nTraining {self.model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Epochs: {n_epochs}, Batch size: {batch_size}, LR: {lr}")
        print(f"Negative samples: {n_negative}, Patience: {patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, n_negative)
            history['train_loss'].append(train_loss)
            
            # Validate
            if (epoch + 1) % 5 == 0 or epoch == 0:
                valid_mrr = self.validate(valid_data, batch_size)
                history['valid_mrr'].append(valid_mrr)
                
                # Update scheduler
                scheduler.step(valid_mrr)
                
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Valid MRR: {valid_mrr:.4f} | "
                      f"Best: {self.best_mrr:.4f}")
                
                # Early stopping and model saving
                if valid_mrr > self.best_mrr:
                    self.best_mrr = valid_mrr
                    self.patience_counter = 0
                    history['best_epoch'] = epoch + 1
                    
                    # Save best model
                    model_path = os.path.join(
                        save_dir, 
                        f"{self.model.__class__.__name__}_best.pt"
                    )
                    torch.save(self.model.state_dict(), model_path)
                    print(f"  → Saved best model (MRR: {valid_mrr:.4f})")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {train_loss:.4f}")
        
        # Load best model
        model_path = os.path.join(
            save_dir, 
            f"{self.model.__class__.__name__}_best.pt"
        )
        self.model.load_state_dict(torch.load(model_path))
        
        # Save training history
        history_path = os.path.join(
            save_dir,
            f"{self.model.__class__.__name__}_history.json"
        )
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ Training completed!")
        print(f"  Best MRR: {self.best_mrr:.4f} at epoch {history['best_epoch']}")
        print(f"  Model saved to: {model_path}")
        
        return history


def load_data(data_dir='processed_data'):
    """Load preprocessed data"""
    train = np.load(os.path.join(data_dir, 'train.npy'))
    valid = np.load(os.path.join(data_dir, 'valid.npy'))
    test = np.load(os.path.join(data_dir, 'test.npy'))
    
    with open(os.path.join(data_dir, 'entity2id.pkl'), 'rb') as f:
        entity2id = pickle.load(f)
    with open(os.path.join(data_dir, 'relation2id.pkl'), 'rb') as f:
        relation2id = pickle.load(f)
    
    return train, valid, test, entity2id, relation2id


def train_model(model_name, train_data, valid_data, n_entities, n_relations,
                embedding_dim=100, n_epochs=100, batch_size=128, lr=0.001,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train a single model
    """
    # Create model
    model = get_model(model_name, n_entities, n_relations, embedding_dim)
    
    # Create trainer
    trainer = KGETrainer(model, n_entities, n_relations, device)
    
    # Train
    history = trainer.train(
        train_data, valid_data,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_negative=1,
        patience=10
    )
    
    return trainer.model, history


if __name__ == "__main__":
    # Configuration
    DATA_DIR = 'processed_data'
    EMBEDDING_DIM = 100
    N_EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_data, valid_data, test_data, entity2id, relation2id = load_data(DATA_DIR)
    
    n_entities = len(entity2id)
    n_relations = len(relation2id)
    
    print(f"Entities: {n_entities}")
    print(f"Relations: {n_relations}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Models to train
    models_to_train = ['TransE', 'TransH', 'RotatE', 'ComplEx', 'CompoundE']
    
    trained_models = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        model, history = train_model(
            model_name, train_data, valid_data,
            n_entities, n_relations,
            embedding_dim=EMBEDDING_DIM,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            device=device
        )
        
        trained_models[model_name] = model
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print("\nRun evaluate_kge.py to evaluate the models.")