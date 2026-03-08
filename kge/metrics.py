"""
Evaluation metrics for Knowledge Graph Embeddings
Includes: Hits@k, MRR, MR, ROC-AUC
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class KGEMetrics:
    """Calculate evaluation metrics for KGE models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def rank_triples(self, test_data, batch_size=128, filtered=False, 
                     all_triples=None):
        """
        Rank test triples against all possible entities
        
        Args:
            test_data: Test triples (h, r, t)
            batch_size: Batch size for evaluation
            filtered: If True, filter out known true triples
            all_triples: Set of all true triples for filtered setting
        
        Returns:
            ranks: Array of ranks for each test triple
            scores_all: All scores for analysis
        """
        self.model.eval()
        n_entities = self.model.n_entities
        
        ranks = []
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                batch_tensor = torch.LongTensor(batch).to(self.device)
                
                for j in range(len(batch)):
                    h, r, t = batch_tensor[j]
                    h_idx, r_idx, t_idx = batch[j]
                    
                    # Score against all possible tails
                    h_repeated = h.repeat(n_entities)
                    r_repeated = r.repeat(n_entities)
                    all_t = torch.arange(n_entities, device=self.device)
                    
                    scores = self.model(h_repeated, r_repeated, all_t)
                    
                    # Filtered setting: set scores of known triples to worst
                    if filtered and all_triples is not None:
                        for entity_id in range(n_entities):
                            if entity_id != t_idx and (h_idx, r_idx, entity_id) in all_triples:
                                scores[entity_id] = float('inf')
                    
                    # Find rank (lower score = better for most models)
                    sorted_indices = torch.argsort(scores)
                    rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
                    
                    ranks.append(rank)
                    all_scores.append(scores.cpu().numpy())
        
        return np.array(ranks), np.array(all_scores)
    
    def calculate_metrics(self, ranks):
        """
        Calculate standard KGE metrics from ranks
        
        Returns:
            dict with MR, MRR, Hits@1, Hits@3, Hits@10
        """
        metrics = {
            'MR': float(np.mean(ranks)),
            'MRR': float(np.mean(1.0 / ranks)),
            'Hits@1': float(np.mean(ranks <= 1)),
            'Hits@3': float(np.mean(ranks <= 3)),
            'Hits@10': float(np.mean(ranks <= 10)),
            'Hits@50': float(np.mean(ranks <= 50)),
        }
        
        return metrics
    
    def calculate_roc_auc(self, test_data, negative_samples, batch_size=128):
        """
        Calculate ROC-AUC score
        
        Args:
            test_data: Positive test triples
            negative_samples: Negative (corrupted) triples
        
        Returns:
            roc_auc: ROC-AUC score
            fpr, tpr: For plotting ROC curve
        """
        self.model.eval()
        
        y_true = []
        y_scores = []
        
        with torch.no_grad():
            # Positive samples
            for i in range(0, len(test_data), batch_size):
                batch = torch.LongTensor(test_data[i:i+batch_size]).to(self.device)
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                scores = self.model(h, r, t)
                
                y_true.extend([1] * len(scores))
                y_scores.extend(scores.cpu().numpy().tolist())
            
            # Negative samples
            for i in range(0, len(negative_samples), batch_size):
                batch = torch.LongTensor(negative_samples[i:i+batch_size]).to(self.device)
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                scores = self.model(h, r, t)
                
                y_true.extend([0] * len(scores))
                y_scores.extend(scores.cpu().numpy().tolist())
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # For models where lower score is better, invert scores
        # This assumes TransE, TransH, RotatE type models
        # For ComplEx, scores are already higher=better
        if hasattr(self.model, 'margin'):
            y_scores = -y_scores
        
        roc_auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        return roc_auc, fpr, tpr
    
    def evaluate(self, test_data, negative_samples=None, batch_size=128, 
                 filtered=False, all_triples=None):
        """
        Complete evaluation pipeline
        
        Returns:
            dict with all metrics
        """
        print("Calculating ranking metrics...")
        ranks, _ = self.rank_triples(
            test_data, 
            batch_size=batch_size,
            filtered=filtered,
            all_triples=all_triples
        )
        
        metrics = self.calculate_metrics(ranks)
        
        # Calculate ROC-AUC if negative samples provided
        if negative_samples is not None:
            print("Calculating ROC-AUC...")
            roc_auc, fpr, tpr = self.calculate_roc_auc(
                test_data, 
                negative_samples, 
                batch_size=batch_size
            )
            metrics['ROC-AUC'] = float(roc_auc)
            metrics['FPR'] = fpr
            metrics['TPR'] = tpr
        
        return metrics


def generate_negative_samples(test_data, n_entities, n_negative=1, seed=42):
    """
    Generate negative samples by corrupting test triples
    
    Args:
        test_data: Test triples
        n_entities: Number of entities
        n_negative: Number of negative samples per positive
    
    Returns:
        negative_samples: Corrupted triples
    """
    np.random.seed(seed)
    negative_samples = []
    
    for h, r, t in test_data:
        for _ in range(n_negative):
            # Randomly corrupt head or tail
            if np.random.rand() < 0.5:
                # Corrupt head
                neg_h = np.random.randint(0, n_entities)
                negative_samples.append([neg_h, r, t])
            else:
                # Corrupt tail
                neg_t = np.random.randint(0, n_entities)
                negative_samples.append([h, r, neg_t])
    
    return np.array(negative_samples)


def plot_metrics_comparison(results_dict, save_path='results/plots/metrics_comparison.png'):
    """
    Plot comparison of metrics across models
    
    Args:
        results_dict: Dict with model names as keys and metrics as values
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    models = list(results_dict.keys())
    metrics_to_plot = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results_dict[model][metric] for model in models]
        
        axes[idx].bar(models, values, color=sns.color_palette("husl", len(models)))
        axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].set_ylim([0, 1.1 * max(values)])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.4f}', 
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison to {save_path}")
    plt.close()


def plot_roc_curves(results_dict, save_path='results/plots/roc_curves.png'):
    """
    Plot ROC curves for all models
    
    Args:
        results_dict: Dict with model names as keys and metrics (including FPR, TPR) as values
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(results_dict))
    
    for idx, (model_name, metrics) in enumerate(results_dict.items()):
        if 'FPR' in metrics and 'TPR' in metrics:
            fpr = metrics['FPR']
            tpr = metrics['TPR']
            roc_auc = metrics['ROC-AUC']
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {save_path}")
    plt.close()


def print_metrics_table(results_dict):
    """
    Print formatted table of metrics
    """
    print("\n" + "="*90)
    print("EVALUATION RESULTS - MODEL COMPARISON")
    print("="*90)
    
    # Header
    header = f"{'Model':<15} {'MR':<10} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10} {'ROC-AUC':<10}"
    print(header)
    print("-"*90)
    
    # Results for each model
    for model_name, metrics in results_dict.items():
        mr = metrics.get('MR', 0)
        mrr = metrics.get('MRR', 0)
        h1 = metrics.get('Hits@1', 0)
        h3 = metrics.get('Hits@3', 0)
        h10 = metrics.get('Hits@10', 0)
        roc_auc = metrics.get('ROC-AUC', 0)
        
        print(f"{model_name:<15} {mr:<10.2f} {mrr:<10.4f} {h1:<10.4f} "
              f"{h3:<10.4f} {h10:<10.4f} {roc_auc:<10.4f}")
    
    print("="*90)
    
    # Find best model for each metric
    print("\nBest Models:")
    print("-" * 90)
    
    best_mrr = max(results_dict.items(), key=lambda x: x[1]['MRR'])
    print(f"  Best MRR: {best_mrr[0]} ({best_mrr[1]['MRR']:.4f})")
    
    best_h10 = max(results_dict.items(), key=lambda x: x[1]['Hits@10'])
    print(f"  Best Hits@10: {best_h10[0]} ({best_h10[1]['Hits@10']:.4f})")
    
    if 'ROC-AUC' in best_mrr[1]:
        best_auc = max(results_dict.items(), key=lambda x: x[1].get('ROC-AUC', 0))
        print(f"  Best ROC-AUC: {best_auc[0]} ({best_auc[1]['ROC-AUC']:.4f})")