"""
Knowledge Graph Embedding Models
Implements: TransE, TransH, RotatE, ComplEx, and CompoundE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    Score function: ||h + r - t||
    """
    def __init__(self, n_entities, n_relations, embedding_dim=100, margin=1.0, norm=2):
        super(TransE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        
        # Entity and relation embeddings
        self.entity_emb = nn.Embedding(n_entities, embedding_dim)
        self.relation_emb = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        
        # Normalize entity embeddings
        self.entity_emb.weight.data = F.normalize(
            self.entity_emb.weight.data, p=2, dim=1
        )
    
    def forward(self, h, r, t):
        """
        Compute score for triples (h, r, t)
        Lower score = more plausible triple
        """
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        
        # TransE: h + r ≈ t
        score = torch.norm(h_emb + r_emb - t_emb, p=self.norm, dim=-1)
        return score
    
    def get_embeddings(self):
        return self.entity_emb.weight.data, self.relation_emb.weight.data


class TransH(nn.Module):
    """
    TransH: Knowledge Graph Embedding by Translating on Hyperplanes
    Projects entities onto relation-specific hyperplanes
    """
    def __init__(self, n_entities, n_relations, embedding_dim=100, margin=1.0, norm=2, c=1.0):
        super(TransH, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        self.c = c  # Soft constraint weight
        
        # Embeddings
        self.entity_emb = nn.Embedding(n_entities, embedding_dim)
        self.relation_emb = nn.Embedding(n_relations, embedding_dim)
        self.relation_norm = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.relation_norm.weight)
    
    def _project(self, emb, norm):
        """Project entity onto hyperplane"""
        return emb - torch.sum(emb * norm, dim=-1, keepdim=True) * norm
    
    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        
        # Normalize relation normal vector
        norm = F.normalize(self.relation_norm(r), p=2, dim=-1)
        
        # Project entities onto hyperplane
        h_proj = self._project(h_emb, norm)
        t_proj = self._project(t_emb, norm)
        
        # Score
        score = torch.norm(h_proj + r_emb - t_proj, p=self.norm, dim=-1)
        return score
    
    def get_embeddings(self):
        return self.entity_emb.weight.data, self.relation_emb.weight.data


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
    Models relations as rotations in complex vector space
    """
    def __init__(self, n_entities, n_relations, embedding_dim=100, margin=1.0):
        super(RotatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = 2.0
        
        # Complex embeddings: real and imaginary parts
        self.entity_emb = nn.Embedding(n_entities, embedding_dim * 2)
        # Relations are phases in complex plane
        self.relation_emb = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.entity_emb.weight)
        # Relation phases between 0 and 2π
        nn.init.uniform_(self.relation_emb.weight, -np.pi, np.pi)
    
    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        
        # Split into real and imaginary parts
        h_re, h_im = torch.chunk(h_emb, 2, dim=-1)
        t_re, t_im = torch.chunk(t_emb, 2, dim=-1)
        
        # Relation as rotation
        r_phase = r_emb / (self.embedding_dim / np.pi)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        
        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        
        # Distance in complex space
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        
        score = torch.norm(torch.cat([diff_re, diff_im], dim=-1), p=2, dim=-1)
        return score
    
    def get_embeddings(self):
        return self.entity_emb.weight.data, self.relation_emb.weight.data


class ComplEx(nn.Module):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction
    Uses complex-valued embeddings with Hermitian dot product
    """
    def __init__(self, n_entities, n_relations, embedding_dim=100):
        super(ComplEx, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        
        # Complex embeddings: real and imaginary parts
        self.entity_re = nn.Embedding(n_entities, embedding_dim)
        self.entity_im = nn.Embedding(n_entities, embedding_dim)
        self.relation_re = nn.Embedding(n_relations, embedding_dim)
        self.relation_im = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.xavier_uniform_(self.relation_re.weight)
        nn.init.xavier_uniform_(self.relation_im.weight)
    
    def forward(self, h, r, t):
        h_re = self.entity_re(h)
        h_im = self.entity_im(h)
        r_re = self.relation_re(r)
        r_im = self.relation_im(r)
        t_re = self.entity_re(t)
        t_im = self.entity_im(t)
        
        # ComplEx scoring: Re(<h, r, conj(t)>)
        score = torch.sum(
            h_re * r_re * t_re +
            h_im * r_re * t_im +
            h_re * r_im * t_im -
            h_im * r_im * t_re,
            dim=-1
        )
        
        return -score  # Negative because higher score = better
    
    def get_embeddings(self):
        # Combine real and imaginary parts
        entity_emb = torch.cat([
            self.entity_re.weight.data,
            self.entity_im.weight.data
        ], dim=1)
        relation_emb = torch.cat([
            self.relation_re.weight.data,
            self.relation_im.weight.data
        ], dim=1)
        return entity_emb, relation_emb


class CompoundE(nn.Module):
    """
    CompoundE: Compound Embedding for Knowledge Graph Completion
    Combines multiple embedding techniques with learnable weights
    Inspired by ensemble methods and multi-view learning
    """
    def __init__(self, n_entities, n_relations, embedding_dim=100, 
                 n_components=3, dropout=0.1):
        super(CompoundE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.margin = 1.0  # Add margin for compatibility
        
        # Multiple embedding spaces (components)
        self.entity_embs = nn.ModuleList([
            nn.Embedding(n_entities, embedding_dim) 
            for _ in range(n_components)
        ])
        self.relation_embs = nn.ModuleList([
            nn.Embedding(n_relations, embedding_dim) 
            for _ in range(n_components)
        ])
        
        # Composition operations for each component
        # 0: Translation (TransE-like)
        # 1: Circular correlation (HolE-like)
        # 2: Element-wise multiplication (DistMult-like)
        
        # Learnable attention weights for combining components
        self.attention = nn.Parameter(torch.ones(n_components) / n_components)
        
        # Projection layers
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        for emb in self.entity_embs:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.relation_embs:
            nn.init.xavier_uniform_(emb.weight)
    
    def _translation_op(self, h, r, t):
        """Translation operation: h + r ≈ t"""
        return torch.norm(h + r - t, p=2, dim=-1)
    
    def _circular_correlation_op(self, h, r, t):
        """Circular correlation (HolE-like)"""
        # Simplified version using element-wise operations
        return torch.norm(self._ccorr(h, r) - t, p=2, dim=-1)
    
    def _ccorr(self, a, b):
        """Circular correlation"""
        a_fft = torch.fft.rfft(a, dim=-1)
        b_fft = torch.fft.rfft(b, dim=-1)
        return torch.fft.irfft(a_fft * torch.conj(b_fft), n=a.shape[-1], dim=-1)
    
    def _multiplication_op(self, h, r, t):
        """Element-wise multiplication (DistMult-like)"""
        score = torch.sum(h * r * t, dim=-1)
        return -score  # Negative for consistency
    
    def forward(self, h, r, t):
        scores = []
        attention_weights = F.softmax(self.attention, dim=0)
        
        for i in range(self.n_components):
            h_emb = self.entity_embs[i](h)
            r_emb = self.relation_embs[i](r)
            t_emb = self.entity_embs[i](t)
            
            # Apply dropout
            h_emb = self.dropout(h_emb)
            r_emb = self.dropout(r_emb)
            t_emb = self.dropout(t_emb)
            
            # Apply projection
            h_emb = self.projection(h_emb)
            t_emb = self.projection(t_emb)
            
            # Choose composition operation based on component
            if i == 0:
                score = self._translation_op(h_emb, r_emb, t_emb)
            elif i == 1:
                score = self._circular_correlation_op(h_emb, r_emb, t_emb)
            else:
                score = self._multiplication_op(h_emb, r_emb, t_emb)
            
            scores.append(score * attention_weights[i])
        
        # Combine scores
        final_score = torch.stack(scores, dim=0).sum(dim=0)
        return final_score
    
    def get_embeddings(self):
        """Return averaged embeddings across components"""
        entity_emb = torch.stack([emb.weight.data for emb in self.entity_embs]).mean(dim=0)
        relation_emb = torch.stack([emb.weight.data for emb in self.relation_embs]).mean(dim=0)
        return entity_emb, relation_emb


def get_model(model_name, n_entities, n_relations, embedding_dim=100, **kwargs):
    """Factory function to create models"""
    models = {
        'TransE': TransE,
        'TransH': TransH,
        'RotatE': RotatE,
        'ComplEx': ComplEx,
        'CompoundE': CompoundE
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](n_entities, n_relations, embedding_dim, **kwargs)