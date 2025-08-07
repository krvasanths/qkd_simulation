import streamlit as st
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load data
edge_features = np.load("edge_features.npy")
edge_index = np.load("edge_index.npy")
node_features = np.load("node_features.npy")

# Parameters
num_edges = edge_features.shape[0]
state_size = edge_features.shape[1]
action_size = 2

# DQN Model
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

# Load model
dqn_model = DQNAgent(state_size, action_size)
dqn_model.load_state_dict(torch.load("dqn_agent_gnn_integrated.pth"))
dqn_model.eval()

# Inference function
def infer_edge_security(edge_feat):
    with torch.no_grad():
        q_values = dqn_model(edge_feat)
        action_probs = F.softmax(q_values, dim=-1)
        predicted_class = torch.argmax(q_values).item()
        return predicted_class, action_probs.numpy()

# NetworkX Graph
def load_network(edge_index, node_features):
    G = nx.Graph()
    for i in range(node_features.shape[0]):
        G.add_node(i)

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        G.add_edge(src, dst, index=i)

    return G

# Streamlit UI
st.set_page_config(page_title="QKD Inference Dashboard", layout="wide")
st.title("Live QKD Security Inference using GNN + DRL")

# Load and display graph
G = load_network(edge_index, node_features)
pos = nx.spring_layout(G, seed=42)

selected_edge_idx = st.slider("Select Edge Index", 0, num_edges - 1, 0)
edge_feat = torch.tensor(edge_features[selected_edge_idx], dtype=torch.float32)
predicted_class, action_probs = infer_edge_security(edge_feat)

# Visual display
st.subheader("DQN Prediction")
st.metric("Predicted Security", "Secure" if predicted_class else "Insecure")
st.text(f"Action Probabilities: {action_probs}")

# Draw graph with highlighted edge
fig, ax = plt.subplots(figsize=(10, 6))
edge_colors = []
for u, v, data in G.edges(data=True):
    idx = data["index"]
    if idx == selected_edge_idx:
        edge_colors.append("red" if predicted_class == 0 else "green")
    else:
        edge_colors.append("gray")

nx.draw(G, pos, ax=ax, with_labels=True, node_color="skyblue", edge_color=edge_colors, node_size=500)
plt.title("QKD Network (Selected Edge Highlighted)")
st.pyplot(fig)
