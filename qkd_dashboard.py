
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
from torch import nn
from torch.nn import functional as F

# ---------------- Load Trained DQN Agent ----------------
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(DQNAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.model(x)

# Load edge features and topology
edge_features = np.load("edge_features.npy")
edge_index = np.load("edge_index.npy")
node_features = np.load("node_features.npy")
with open("qkd_topology.gpickle", "rb") as f:
    G = pickle.load(f)

edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float32)

# Load trained model
agent = DQNAgent(state_size=edge_features.shape[1], action_size=2)
agent.load_state_dict(torch.load("dqn_agent_gnn_integrated.pth"))
agent.eval()

# ---------------- Streamlit Dashboard ----------------
st.set_page_config(layout="wide")
st.title("🔐 Quantum Key Distribution (QKD) Routing Simulator with DRL")

# Sidebar controls
nodes = list(G.nodes())
source = st.sidebar.selectbox("Select Source Node", nodes)
target = st.sidebar.selectbox("Select Target Node", nodes, index=len(nodes) - 1)
eavesdrop = st.sidebar.toggle("💀 Simulate Eavesdropping")

# Recalculate QBER if eavesdrop enabled
for u, v in G.edges():
    if eavesdrop:
        G[u][v]['qber'] = min(1.0, G[u][v]['qber'] + 0.1)
    else:
        original_qber = G[u][v].get('original_qber', G[u][v]['qber'])
        G[u][v]['qber'] = original_qber

# Predict edge security using DQN agent
edge_security = {}
with torch.no_grad():
    for idx, (u, v) in enumerate(G.edges()):
        edge_feat = edge_attr_tensor[idx]
        q_vals = agent(edge_feat)
        action = torch.argmax(q_vals).item()
        edge_security[(u, v)] = action  # 0: Insecure, 1: Secure

# Assign color to edges
edge_colors = []
for u, v in G.edges():
    if edge_security.get((u, v), 0) == 1:
        edge_colors.append("green")
    else:
        edge_colors.append("red")

# Path finding on secure edges only
secure_G = G.edge_subgraph([e for e in G.edges() if edge_security.get(e, 0) == 1]).copy()
try:
    path = nx.shortest_path(secure_G, source=source, target=target)
except nx.NetworkXNoPath:
    path = []

# Draw network
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, edge_color=edge_colors, node_color='skyblue', with_labels=True, node_size=500)
if path:
    edges_in_path = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='blue', width=3)
    st.success(f"DRL-Optimized Secure Path: {path}")
else:
    st.error("No secure path found between source and target.")

st.pyplot(plt)
