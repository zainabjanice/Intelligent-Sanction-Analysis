import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# python-louvain can be available as community.community_louvain or community
import importlib

# try dynamic import to avoid static import resolution issues in linters
community_louvain = None
for _mod in ("community.community_louvain", "community", "community_louvain"):
    try:
        community_louvain = importlib.import_module(_mod)
        break
    except ModuleNotFoundError:
        continue

# ========================
# GRAPH ANALYTICS FUNCTIONS
# ========================

import networkx as nx

def compute_graph_statistics(G):
    stats = {
        "Number of nodes": G.number_of_nodes(),
        "Number of edges": G.number_of_edges(),
        "Graph type": "Directed" if G.is_directed() else "Undirected",
    }

    if G.is_directed():
        stats["Weakly connected"] = nx.is_weakly_connected(G)
        stats["Strongly connected"] = nx.is_strongly_connected(G)
        stats["Number of weakly connected components"] = nx.number_weakly_connected_components(G)
    else:
        stats["Connected"] = nx.is_connected(G)
        stats["Number of connected components"] = nx.number_connected_components(G)

    return stats


def pagerank_analysis(G, top_k=15):
    pagerank = nx.pagerank(G, alpha=0.85)
    pagerank_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    results = []
    for name, score in pagerank_sorted[:top_k]:
        results.append({
            "name": name,
            "pagerank": score,
            "degree": G.degree(name)
        })
    return pd.DataFrame(results)

def community_detection(G):
    if community_louvain is None:
        raise ImportError("Install python-louvain: pip install python-louvain")
    # best_partition expects an undirected graph
    if G.is_directed():
        G_for_partition = G.to_undirected()
    else:
        G_for_partition = G
    partition = community_louvain.best_partition(G_for_partition)
    communities = {}
    for node, cid in partition.items():
        if cid not in communities:
            communities[cid] = []
        communities[cid].append(node)
    results = []
    for cid, members in communities.items():
        results.append({
            "community_id": cid,
            "size": len(members),
            "members_sample": ", ".join(members[:3])
        })
    return pd.DataFrame(results), partition

def centrality_analysis(G, top_k=10):
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    results = []
    for node in G.nodes():
        results.append({
            "node": node,
            "betweenness": betweenness[node],
            "closeness": closeness[node],
            "degree_centrality": degree_centrality[node],
            "degree": G.degree(node)
        })
    df = pd.DataFrame(results)
    return df.sort_values("betweenness", ascending=False).head(top_k)

def anomaly_detection(G, z_threshold=2):
    degrees = np.array([G.degree(n) for n in G.nodes()])
    mean = degrees.mean()
    std = degrees.std()
    anomalies = []
    for n in G.nodes():
        z = abs((G.degree(n) - mean)/std) if std>0 else 0
        if z > z_threshold:
            anomalies.append({"node": n, "degree": G.degree(n), "z_score": round(z,2)})
    return pd.DataFrame(anomalies)
