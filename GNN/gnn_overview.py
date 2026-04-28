# =============================================
# Clean GNN Implementation (BO-ready + WSC aligned)
# =============================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.loader import DataLoader
import random 

# -------------------------
# Global settings
# -------------------------
DATASET_PATH = "./data"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------
# Dataset loader 
# -------------------------
def load_dataset(name="MUTAG", data_seed=42):
    dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name=name)

    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(len(dataset))

    n_total = len(dataset)
    train_end = int(0.8 * n_total)
    val_end = int(0.9 * n_total)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return dataset, train_loader, val_loader, test_loader


# -------------------------
# GNN backbone
# -------------------------
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, num_layers=2, layer_name="GCN", dp_rate=0.1):
        super().__init__()

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels = c_in

        for _ in range(num_layers - 1):
            layers.append(gnn_layer(in_channels, c_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dp_rate))
            in_channels = c_hidden

        layers.append(gnn_layer(in_channels, c_hidden))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


# -------------------------
# Graph-level model
# -------------------------
class GraphGNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=3,
        layer_name="GraphConv",
        dp_rate=0.0,
        dp_rate_linear=0.5,
    ):
        super().__init__()

        self.gnn = GNNModel(
            c_in=c_in,
            c_hidden=c_hidden,
            num_layers=num_layers,
            layer_name=layer_name,
            dp_rate=dp_rate,
        )

        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out),
        )

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch)
        x = self.head(x)
        return x


# -------------------------
# Training + Evaluation
# -------------------------
def train_graph_classifier(
    model_name="GraphConv",
    c_hidden=64,
    layer_name="GraphConv",
    num_layers=3,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    dataset="MUTAG",
    seed=0,
    data_seed=42,   
    epochs=80,
    lr=1e-2,
):
    """
    BO-compatible training function

    Returns:
        model, {"train": acc, "val": acc, "test": acc}
    """

    # ========= seed =========
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= dataset =========
    dataset_obj, train_loader, val_loader, test_loader = load_dataset(dataset, data_seed)

    # ========= model =========
    model = GraphGNNModel(
        c_in=dataset_obj.num_node_features,
        c_hidden=c_hidden,
        c_out=1 if dataset_obj.num_classes == 2 else dataset_obj.num_classes,
        num_layers=num_layers,
        layer_name=layer_name,
        dp_rate=dp_rate,
        dp_rate_linear=dp_rate_linear,
    ).to(device)

    # ========= optimizer =========
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ========= loss =========
    if dataset_obj.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ========= training =========
    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            if dataset_obj.num_classes == 2:
                out = out.view(-1)

            if dataset_obj.num_classes == 2:
                loss = criterion(out, batch.y.float())
            else:
                loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

    # ========= evaluation =========
    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)

                if dataset_obj.num_classes == 2:
                    preds = (out.view(-1) > 0).long()
                else:
                    preds = out.argmax(dim=-1)

                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

        return correct / total

    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)

    return model, {
        "train": float(train_acc),
        "val": float(val_acc),
        "test": float(test_acc),
    }


# -------------------------
# Debug run
# -------------------------
if __name__ == "__main__":
    model, result = train_graph_classifier(
        model_name="GraphConv",
        c_hidden=256,
        num_layers=3,
        dp_rate_linear=0.5,
        dp_rate=0.0,
        dataset="MUTAG",
    )

    print(result)
