import streamlit as st
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

# Define DQN model
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Load files
edge_features = np.load("edge_features.npy")
edge_index = np.load("edge_index.npy")
node_features = np.load("node_features.npy")

num_nodes = node_features.shape[0]
num_edges = edge_index.shape[1]
state_size = edge_features.shape[1]
action_size = 2  # secure or insecure

# Load trained DQN agent
agent = DQNAgent(state_size, action_size)
agent.load_state_dict(torch.load("dqn_agent_gnn_integrated.pth"))
agent.eval()

def build_graph_from_gpickle(gpickle_path="qkd_topology.gpickle", agent=None):
    G = nx.read_gpickle(gpickle_path)
    for u, v, data in G.edges(data=True):
        # Build edge feature vector
        edge_feat = torch.tensor([
            data.get("distance_km", 0),
            data.get("eta_ch", 0),
            data.get("qber", 0),
            data.get("key_rate", 0)
        ], dtype=torch.float32)

        if agent:
            with torch.no_grad():
                q_vals = agent(edge_feat)
                secure_prob = F.softmax(q_vals, dim=0)[1].item()  # Class 1 = secure
        else:
            secure_prob = 0.5  # default probability

        data["edge_feat"] = edge_feat
        data["secure_score"] = secure_prob

    return G
    
# Build networkx graph
def build_graph(edge_index, edge_features):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        feat = torch.tensor(edge_features[i], dtype=torch.float32)
        pred = agent(feat)
        score = F.softmax(pred, dim=0)[1].item()  # Secure probability
        G.add_edge(u, v, index=i, secure_score=score)
    return G

# Find most secure path (maximize security scores)
def find_secure_path(G, source, target, eavesdrop=False):
    edge_weights = {
        (u, v): (1 - G[u][v]['secure_score']) if not eavesdrop else (1 - G[u][v]['secure_score'] * 0.5)
        for u, v in G.edges()
    }
    st.success(edge_weights)
    nx.set_edge_attributes(G, edge_weights, "weight")
    try:
        return nx.shortest_path(G, source=source, target=target, weight="weight")
    except nx.NetworkXNoPath:
        return []

# UI
st.set_page_config("QKD Routing with DQN", layout="wide")
st.title("QKD Routing Optimizer using GNN + DQN Agent")

col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("Select Source Node", list(range(num_nodes)), index=0)
with col2:
    target = st.selectbox("Select Target Node", list(range(num_nodes)), index=min(num_nodes - 1, 10))

eavesdrop = st.toggle("Simulate Eavesdropping", value=False)

# Build graph and path
# G = build_graph(edge_index, edge_features)
G = build_graph_from_gpickle()
path = find_secure_path(G, source, target, eavesdrop)

# Plot
st.subheader("Optimized Path")
if path:
    st.success(f"Path: {' → '.join(map(str, path))}")
else:
    st.error("No secure path found!")

# Visualize
pos = nx.spring_layout(G, seed=42)
edge_colors = []
edge_widths = []
for u, v, data in G.edges(data=True):
    idx = data["index"]
    if path and (u, v) in zip(path, path[1:]) or (v, u) in zip(path, path[1:]):
        edge_colors.append("green")
        edge_widths.append(3)
    elif data["secure_score"] > 0.5:
        edge_colors.append("gray")
        edge_widths.append(1)
    else:
        edge_colors.append("red")
        edge_widths.append(1)

fig, ax = plt.subplots(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color=edge_colors, width=edge_widths, node_size=500)
plt.title("QKD Network with DQN-Inferred Secure Routing")
st.pyplot(fig)
