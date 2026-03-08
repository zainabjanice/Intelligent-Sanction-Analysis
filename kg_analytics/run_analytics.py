from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import networkx as nx
import importlib
import os
import sys
import pandas as pd

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# DYNAMIC IMPORT OF GRAPH ANALYTICS MODULE
# =====================================================

def load_graph_analytics():
    """
    Dynamically import kg_analytics.graph_analytics
    Ensures correct local module is loaded (not site-packages).
    """
    try:
        mod = importlib.import_module("kg_analytics.graph_analytics")
    except ModuleNotFoundError:
        project_root = os.path.dirname(BASE_DIR)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        mod = importlib.import_module("kg_analytics.graph_analytics")

    return (
        mod.compute_graph_statistics,
        mod.pagerank_analysis,
        mod.community_detection,
        mod.centrality_analysis,
        mod.anomaly_detection,
    )


(
    compute_graph_statistics,
    pagerank_analysis,
    community_detection,
    centrality_analysis,
    anomaly_detection,
) = load_graph_analytics()

# =====================================================
# NEO4J CONNECTION
# =====================================================

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "12345678"  # move to .env later

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# =====================================================
# LOAD GRAPH
# =====================================================

def load_graph_from_neo4j():
    """
    Load Knowledge Graph from Neo4j into a NetworkX DiGraph.
    """
    G = nx.DiGraph()

    query = """
    MATCH (p:Person)-[r]->(n)
    RETURN p.name AS person,
           type(r) AS relation,
           n.name AS target
    """

    with driver.session() as session:
        for record in session.run(query):
            if record["person"] and record["target"]:
                G.add_node(record["person"], label="Person")
                G.add_node(record["target"], label="Entity")
                G.add_edge(
                    record["person"],
                    record["target"],
                    relation=record["relation"]
                )

    return G

# -------------------------------
# Degree Distribution Plot
# -------------------------------
def save_degree_distribution(G, output_path):
        degrees = [d for _, d in G.degree()]
        
        plt.figure()
        plt.hist(degrees, bins=50)
        plt.xlabel("Degree")
        plt.ylabel("Number of Nodes")
        plt.title("Degree Distribution of Interpol Knowledge Graph")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

#-------------------------------
# Add PageRank plot TOP-K
def save_pagerank_plot(pagerank_df, output_path):
        plt.figure()
        plt.barh(pagerank_df["name"], pagerank_df["pagerank"])
        plt.xlabel("PageRank Score")
        plt.ylabel("Node")
        plt.title("Top-10 PageRank Nodes")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


# =====================================================
# RUN ANALYTICS
# =====================================================

if __name__ == "__main__":
    G = load_graph_from_neo4j()

    # -------------------------------
    # Graph statistics
    # -------------------------------
    stats = compute_graph_statistics(G)
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(RESULTS_DIR, "graph_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"✔ Graph statistics saved → {stats_path}")

    # -------------------------------
    # PageRank
    # -------------------------------
    pr_df = pagerank_analysis(G, top_k=10)
    pr_path = os.path.join(RESULTS_DIR, "pagerank_top10.csv")
    pr_df.to_csv(pr_path, index=False)
    print(f"✔ PageRank saved → {pr_path}")

    # -------------------------------
    # Communities
    # -------------------------------
    communities_df, _ = community_detection(G)
    comm_path = os.path.join(RESULTS_DIR, "communities.csv")
    communities_df.to_csv(comm_path, index=False)
    print(f"✔ Communities saved → {comm_path}")

    # -------------------------------
    # Centrality
    # -------------------------------
    cent_df = centrality_analysis(G, top_k=10)
    cent_path = os.path.join(RESULTS_DIR, "centrality_top10.csv")
    cent_df.to_csv(cent_path, index=False)
    print(f"✔ Centrality saved → {cent_path}")

    # -------------------------------
    # Anomalies
    # -------------------------------
    anomaly_df = anomaly_detection(G)
    anomaly_path = os.path.join(RESULTS_DIR, "anomalies.csv")
    anomaly_df.to_csv(anomaly_path, index=False)
    print(f"✔ Anomalies saved → {anomaly_path}")

    # -------------------------------
    # Degree Distribution Plot
    # -------------------------------
    degree_plot_path = os.path.join(PLOTS_DIR, "degree_distribution.png")
    save_degree_distribution(G, degree_plot_path)
    print(f"✔ Degree distribution plot saved → {degree_plot_path}")

    # -------------------------------
    # PageRank Plot
    # -------------------------------
    pagerank_plot_path = os.path.join(PLOTS_DIR, "pagerank.png")
    save_pagerank_plot(pr_df, pagerank_plot_path)
    print(f"✔ PageRank plot saved → {pagerank_plot_path}")


    driver.close()
