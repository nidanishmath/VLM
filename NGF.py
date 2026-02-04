import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Edge Updation
class EdgeUpdateNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, h_v_prev, h_u_prev, e_vu_prev):
        return self.mlp(torch.cat([h_v_prev, h_u_prev, e_vu_prev], dim=-1))

# Downstream Node Updation
class DownstreamNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(3 * dim, dim)

    def forward(self, h0_v, h_prev_v, e_vu):
        return self.fc(torch.cat([h0_v, h_prev_v, e_vu], dim=-1))

# Upstream Node Updation
class UpstreamNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(3 * dim, dim)

    def forward(self, h0_v, h_prev_v, e_vu):
        return self.fc(torch.cat([h0_v, h_prev_v, e_vu], dim=-1))

# Node Updation
class NodeUpdateNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, h_ds, h_us):
        return self.fc(torch.cat([h_ds, h_us], dim=-1))

# Distance Proxy
def node_distance(h):
    return torch.norm(h, p=2)

# Convergence Criterion
def has_converged(prev_feats, curr_feats, eps=1e-4):
    diffs = []
    for v in curr_feats:
        diffs.append(torch.norm(curr_feats[v] - prev_feats[v]))
    return torch.mean(torch.stack(diffs)) < eps

# NGF
def NGF_scheme(
    A,
    P,
    h0,
    maxIteration=5,
    device="cuda"
):
    """
    A : adjacency network
    P : list of nodes
    h0: initial node features {v: tensor}
    """

    dim = next(iter(h0.values())).shape[-1]

    Ne = EdgeUpdateNet(dim).to(device)
    NDs = DownstreamNet(dim).to(device)
    NUs = UpstreamNet(dim).to(device)
    Nn = NodeUpdateNet(dim).to(device)

    # initialize
    k = 1
    hk_prev = {v: h0[v].clone() for v in P}

    while True:
        if k >= maxIteration:
            break

        hk_curr = {}

        for v in P:
            Psi_v = A["adjacency"][v]

            hDs_v = torch.zeros(dim, device=device)
            hUs_v = torch.zeros(dim, device=device)

            for u in Psi_v:
                # ---- Edge update ----
                e_prev = A["edge_features"][(v, u)]
                e_k = Ne(
                    hk_prev[v],
                    hk_prev[u],
                    e_prev
                )

                A["edge_features"][(v, u)] = e_k

                # ---- Directional aggregation ----
                if node_distance(hk_prev[u]) < node_distance(hk_prev[v]):
                    hDs_v += NDs(h0[v], hk_prev[v], e_k)

                if node_distance(hk_prev[u]) > node_distance(hk_prev[v]):
                    hUs_v += NUs(h0[v], hk_prev[v], e_k)

            # ---- Node update ----
            hk_curr[v] = Nn(hDs_v, hUs_v)

            A["node_features"][v] = hk_curr[v]

        # ---- Convergence check ----
        if has_converged(hk_prev, hk_curr):
            break

        hk_prev = hk_curr
        k += 1

    return A

# Compute centroids
def compute_centroids(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    centroids = {}
    for sp_id in np.unique(labels):
        if sp_id < 0:
            continue
        ys, xs = np.where(labels == sp_id)
        centroids[int(sp_id)] = (int(xs.mean()), int(ys.mean()))
    return centroids

# Visualization
def visualize_A_updated(
    image_path,
    labels,
    A,
    savename,
    node_radius=3,
    edge_thickness=1,
    node_color=(0, 255, 255),   # yellow
    edge_color=(255, 255, 0)    # cyan
):
    image = cv2.imread(image_path)

    # Safe conversion
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    centroids = compute_centroids(labels)
    vis = image.copy()

    # ---- Draw edges ----
    for v, neighbors in A["adjacency"].items():
        if v not in centroids:
            continue
        x1, y1 = centroids[v]

        for u in neighbors:
            if u not in centroids:
                continue
            x2, y2 = centroids[u]

            cv2.line(
                vis,
                (x1, y1),
                (x2, y2),
                edge_color,
                edge_thickness
            )

    # ---- Draw nodes ----
    for v, (x, y) in centroids.items():
        cv2.circle(
            vis,
            (x, y),
            node_radius,
            node_color,
            -1
        )

    # ---- Save ----
    cv2.imwrite(savename, vis)

    # ---- Show inline ----
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Updated Adjacency Network A (after NGF)")
    plt.axis("off")
    plt.show()

    return vis

