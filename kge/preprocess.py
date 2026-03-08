"""
Preprocess exported Neo4j triples for KGE training
"""
import os
import numpy as np
import pickle
from collections import defaultdict


class KGPreprocessor:
    """Preprocess knowledge graph triples for embedding models"""
    
    def __init__(self, triples_file):
        self.triples_file = triples_file
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
    
    def load_triples(self):
        """Load triples from TSV file"""
        triples = []
        print(f"\nLoading triples from {self.triples_file}...")
        
        with open(self.triples_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append(parts)
                else:
                    print(f"⚠ Warning: Skipping malformed line {line_num}: {line.strip()}")
        
        print(f"✓ Loaded {len(triples)} triples")
        return triples
    
    def build_mappings(self, triples):
        """Build entity and relation to ID mappings"""
        print("\nBuilding entity and relation mappings...")
        
        entities = set()
        relations = set()
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        # Create sorted mappings for consistency
        self.entity2id = {e: idx for idx, e in enumerate(sorted(entities))}
        self.relation2id = {r: idx for idx, r in enumerate(sorted(relations))}
        
        # Create reverse mappings
        self.id2entity = {idx: e for e, idx in self.entity2id.items()}
        self.id2relation = {idx: r for r, idx in self.relation2id.items()}
        
        print(f"✓ Number of unique entities: {len(self.entity2id)}")
        print(f"✓ Number of unique relations: {len(self.relation2id)}")
        
        # Print relation types
        print("\nRelation types found:")
        for rel in sorted(relations):
            print(f"  - {rel}")
    
    def convert_to_ids(self, triples):
        """Convert triples to ID format"""
        print("\nConverting triples to ID format...")
        
        id_triples = []
        for h, r, t in triples:
            id_triples.append([
                self.entity2id[h],
                self.relation2id[r],
                self.entity2id[t]
            ])
        
        return np.array(id_triples, dtype=np.int64)
    
    def split_data(self, triples, train_ratio=0.8, valid_ratio=0.1, seed=42):
        """
        Split data into train/valid/test sets
        
        Args:
            triples: Array of triples
            train_ratio: Ratio of training data
            valid_ratio: Ratio of validation data
            seed: Random seed for reproducibility
        """
        print("\nSplitting data...")
        
        np.random.seed(seed)
        n_triples = len(triples)
        
        # Shuffle indices
        indices = np.random.permutation(n_triples)
        
        # Calculate split points
        train_size = int(n_triples * train_ratio)
        valid_size = int(n_triples * valid_ratio)
        
        # Split indices
        train_idx = indices[:train_size]
        valid_idx = indices[train_size:train_size + valid_size]
        test_idx = indices[train_size + valid_size:]
        
        # Create splits
        train = triples[train_idx]
        valid = triples[valid_idx]
        test = triples[test_idx]
        
        print(f"✓ Train: {len(train)} triples ({train_ratio*100:.0f}%)")
        print(f"✓ Valid: {len(valid)} triples ({valid_ratio*100:.0f}%)")
        print(f"✓ Test:  {len(test)} triples ({(1-train_ratio-valid_ratio)*100:.0f}%)")
        
        return train, valid, test
    
    def get_statistics(self, triples, split_name="Dataset"):
        """Print statistics about the triples"""
        print(f"\n{split_name} Statistics:")
        print("-" * 50)
        
        # Count unique entities and relations
        entities = set()
        relations = defaultdict(int)
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations[self.id2relation[r]] += 1
        
        print(f"  Unique entities: {len(entities)}")
        print(f"  Unique relations: {len(relations)}")
        print(f"  Total triples: {len(triples)}")
        print(f"  Avg triples per relation: {len(triples) / len(relations):.2f}")
        
        # Top relations
        print("\n  Top 5 relations by frequency:")
        for rel, count in sorted(relations.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {rel}: {count} ({count/len(triples)*100:.1f}%)")
    
    def save_processed_data(self, output_dir='processed_data', 
                           train_ratio=0.8, valid_ratio=0.1):
        """
        Process and save all data
        
        Args:
            output_dir: Directory to save processed data
            train_ratio: Ratio of training data
            valid_ratio: Ratio of validation data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process
        triples = self.load_triples()
        
        if len(triples) == 0:
            raise ValueError("No triples loaded! Check your input file.")
        
        self.build_mappings(triples)
        id_triples = self.convert_to_ids(triples)
        
        # Split data
        train, valid, test = self.split_data(id_triples, train_ratio, valid_ratio)
        
        # Save splits
        print(f"\nSaving processed data to {output_dir}/...")
        np.save(os.path.join(output_dir, 'train.npy'), train)
        np.save(os.path.join(output_dir, 'valid.npy'), valid)
        np.save(os.path.join(output_dir, 'test.npy'), test)
        print("✓ Saved train.npy, valid.npy, test.npy")
        
        # Save mappings
        mappings = {
            'entity2id': self.entity2id,
            'relation2id': self.relation2id,
            'id2entity': self.id2entity,
            'id2relation': self.id2relation
        }
        
        for name, mapping in mappings.items():
            path = os.path.join(output_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(mapping, f)
        print("✓ Saved entity2id.pkl, relation2id.pkl, id2entity.pkl, id2relation.pkl")
        
        # Print statistics
        self.get_statistics(train, "Training Set")
        self.get_statistics(valid, "Validation Set")
        self.get_statistics(test, "Test Set")
        
        # Save summary
        summary = {
            'n_entities': len(self.entity2id),
            'n_relations': len(self.relation2id),
            'n_triples': len(id_triples),
            'n_train': len(train),
            'n_valid': len(valid),
            'n_test': len(test)
        }
        
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Knowledge Graph Summary\n")
            f.write("=" * 50 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        print(f"\n✓ Saved summary to {summary_path}")
        
        return train, valid, test


if __name__ == "__main__":
    # Configuration
    TRIPLES_FILE = 'data/triples.tsv'
    OUTPUT_DIR = 'processed_data'
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    
    print("="*60)
    print("KNOWLEDGE GRAPH PREPROCESSING")
    print("="*60)
    
    # Check input file
    if not os.path.exists(TRIPLES_FILE):
        print(f"\n❌ Error: Input file not found: {TRIPLES_FILE}")
        print("\nPlease run export_triples.py first to export triples from Neo4j")
        exit(1)
    
    # Preprocess
    preprocessor = KGPreprocessor(TRIPLES_FILE)
    
    try:
        train, valid, test = preprocessor.save_processed_data(
            output_dir=OUTPUT_DIR,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO
        )
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nProcessed data saved to: {OUTPUT_DIR}/")
        print("\nNext steps:")
        print("  1. Run train_kge.py to train embedding models")
        print("  2. Run evaluate_kge.py to evaluate trained models")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        raise