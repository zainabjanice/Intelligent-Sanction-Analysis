"""
Evaluation script for trained Knowledge Graph Embedding models
"""
import os
import sys
import pickle
import numpy as np
import torch
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kge.models import get_model
    from kge.metrics import (
        KGEMetrics, 
        generate_negative_samples,
        plot_metrics_comparison,
        plot_roc_curves,
        print_metrics_table
    )
except ImportError:
    from models import get_model
    from metrics import (
        KGEMetrics, 
        generate_negative_samples,
        plot_metrics_comparison,
        plot_roc_curves,
        print_metrics_table
    )


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


def load_trained_model(model_name, n_entities, n_relations, embedding_dim=100,
                       model_dir='results/embeddings', device='cpu'):
    """Load a trained model"""
    model = get_model(model_name, n_entities, n_relations, embedding_dim)
    model_path = os.path.join(model_dir, f"{model_name}_best.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model_name, model, test_data, n_entities, 
                   batch_size=128, device='cpu', calculate_auc=True):
    """
    Evaluate a single model
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    evaluator = KGEMetrics(model, device)
    
    # Generate negative samples for ROC-AUC
    negative_samples = None
    if calculate_auc:
        print("Generating negative samples for ROC-AUC...")
        negative_samples = generate_negative_samples(test_data, n_entities, n_negative=1)
    
    # Evaluate
    metrics = evaluator.evaluate(
        test_data,
        negative_samples=negative_samples,
        batch_size=batch_size,
        filtered=False
    )
    
    print(f"\nResults for {model_name}:")
    print(f"  MR:        {metrics['MR']:.2f}")
    print(f"  MRR:       {metrics['MRR']:.4f}")
    print(f"  Hits@1:    {metrics['Hits@1']:.4f}")
    print(f"  Hits@3:    {metrics['Hits@3']:.4f}")
    print(f"  Hits@10:   {metrics['Hits@10']:.4f}")
    if 'ROC-AUC' in metrics:
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    return metrics


def save_results(results, output_dir='results/metrics'):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results for JSON (remove numpy arrays)
    json_results = {}
    for model_name, metrics in results.items():
        json_results[model_name] = {
            k: float(v) if not isinstance(v, (list, np.ndarray)) else None
            for k, v in metrics.items()
        }
    
    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {json_path}")
    
    # Save complete results (including arrays) as pickle
    pickle_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Complete results saved to: {pickle_path}")


def save_embeddings(models, entity2id, relation2id, output_dir='results/embeddings'):
    """Save learned embeddings for visualization and analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        entity_emb, relation_emb = model.get_embeddings()
        
        embeddings = {
            'entity_embeddings': entity_emb.cpu().numpy(),
            'relation_embeddings': relation_emb.cpu().numpy(),
            'entity2id': entity2id,
            'relation2id': relation2id
        }
        
        emb_path = os.path.join(output_dir, f'{model_name}_embeddings.pkl')
        with open(emb_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"✓ Saved {model_name} embeddings to: {emb_path}")


def main():
    """Main evaluation pipeline"""
    # Configuration
    DATA_DIR = 'processed_data'
    MODEL_DIR = 'results/embeddings'
    METRICS_DIR = 'results/metrics'
    PLOTS_DIR = 'results/plots'
    EMBEDDING_DIM = 100
    BATCH_SIZE = 128
    CALCULATE_AUC = True
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_data, valid_data, test_data, entity2id, relation2id = load_data(DATA_DIR)
    
    n_entities = len(entity2id)
    n_relations = len(relation2id)
    
    print(f"Entities: {n_entities}")
    print(f"Relations: {n_relations}")
    print(f"Test triples: {len(test_data)}")
    
    # Models to evaluate
    model_names = ['TransE', 'TransH', 'RotatE', 'ComplEx', 'CompoundE']
    
    results = {}
    models = {}
    
    # Evaluate each model
    for model_name in model_names:
        try:
            # Load model
            model = load_trained_model(
                model_name, n_entities, n_relations,
                embedding_dim=EMBEDDING_DIM,
                model_dir=MODEL_DIR,
                device=device
            )
            
            # Evaluate
            metrics = evaluate_model(
                model_name, model, test_data, n_entities,
                batch_size=BATCH_SIZE,
                device=device,
                calculate_auc=CALCULATE_AUC
            )
            
            results[model_name] = metrics
            models[model_name] = model
            
        except FileNotFoundError as e:
            print(f"\n⚠ Warning: {e}")
            print(f"  Skipping {model_name}")
            continue
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            continue
    
    if not results:
        print("\n❌ No models were successfully evaluated!")
        return
    
    # Print comparison table
    print_metrics_table(results)
    
    # Save results
    save_results(results, METRICS_DIR)
    
    # Save embeddings
    print("\nSaving embeddings...")
    save_embeddings(models, entity2id, relation2id, MODEL_DIR)
    
    # Generate plots
    print("\nGenerating plots...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    try:
        plot_metrics_comparison(results, 
            save_path=os.path.join(PLOTS_DIR, 'metrics_comparison.png'))
    except Exception as e:
        print(f"⚠ Warning: Could not generate metrics plot: {e}")
    
    try:
        if CALCULATE_AUC:
            plot_roc_curves(results, 
                save_path=os.path.join(PLOTS_DIR, 'roc_curves.png'))
    except Exception as e:
        print(f"⚠ Warning: Could not generate ROC plot: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results directory: results/")
    print(f"  - Metrics: {METRICS_DIR}")
    print(f"  - Embeddings: {MODEL_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['MRR'])
    print(f"\n🏆 Best model: {best_model[0]} (MRR: {best_model[1]['MRR']:.4f})")


if __name__ == "__main__":
    main()