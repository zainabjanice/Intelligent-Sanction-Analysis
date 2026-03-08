"""
Export triples from Neo4j knowledge graph for KGE training
"""
from neo4j import GraphDatabase
import os

class TripleExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def export_triples(self, output_file="data/triples.tsv"):
        """
        Export all triples from Neo4j in format: head\trelation\ttail
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (h)-[r]->(t)
                RETURN labels(h)[0] + ":" + h.name AS head,
                       type(r) AS relation,
                       labels(t)[0] + ":" + t.name AS tail
            """)
            
            with open(output_file, "w", encoding="utf-8") as f:
                count = 0
                for row in result:
                    f.write(f"{row['head']}\t{row['relation']}\t{row['tail']}\n")
                    count += 1
        
        print(f"✓ Exported {count} triples to {output_file}")
        return count
    
    def get_statistics(self):
        """Get KG statistics"""
        with self.driver.session() as session:
            # Count entities
            entity_result = session.run("""
                MATCH (n)
                RETURN count(DISTINCT n) as entity_count,
                       count(DISTINCT labels(n)[0]) as entity_types
            """)
            entities = entity_result.single()
            
            # Count relations
            relation_result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as triple_count,
                       count(DISTINCT type(r)) as relation_types
            """)
            relations = relation_result.single()
            
            stats = {
                'entities': entities['entity_count'],
                'entity_types': entities['entity_types'],
                'triples': relations['triple_count'],
                'relation_types': relations['relation_types']
            }
            
            print("\n=== Knowledge Graph Statistics ===")
            print(f"Total Entities: {stats['entities']}")
            print(f"Entity Types: {stats['entity_types']}")
            print(f"Total Triples: {stats['triples']}")
            print(f"Relation Types: {stats['relation_types']}")
            print("=" * 35)
            
            return stats


if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    OUTPUT_FILE = "data/triples.tsv"
    
    # Export triples
    exporter = TripleExporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Get statistics
        exporter.get_statistics()
        
        # Export triples
        exporter.export_triples(OUTPUT_FILE)
        
    finally:
        exporter.close()