import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Edge Updation
class EdgeUpdateNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3*d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, h_v, h_u, e_vu):
        return self.net(torch.cat([h_v, h_u, e_vu], dim=-1))

# Downstream Node Updation
class DownstreamNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(3*d, d)

    def forward(self, h0_v, h_prev_v, e_vu):
        return self.net(torch.cat([h0_v, h_prev_v, e_vu], dim=-1))

# Upstream Node Updation
class UpstreamNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(3 * dim, dim)

    def forward(self, h0_v, h_prev_v, e_vu):
        return self.fc(torch.cat([h0_v, h_prev_v, e_vu], dim=-1))

# Node Updation
class NodeUpdateNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, h_ds, h_us):
        return self.net(torch.cat([h_ds, h_us], dim=-1))

# Distance Proxy
def path_distance(v, P_index):
    return P_index[v]

# Adjacency Network
def adjacency_matrix(A, P):
    idx = {v:i for i,v in enumerate(P)}
    n = len(P)
    M = torch.zeros((n, n), dtype=torch.int32)

    for v in P:
        i = idx[v]
        for u in A["adjacency"][v]:
            if u in idx:
                j = idx[u]
                M[i, j] = 1
    return M

# Convergence check
def has_converged_adj(A_history, km):
    if len(A_history) < km + 1:
        return False

    ne = A_history[-1].sum().item()
    if ne == 0:
        return True

    sigma = 0
    for j in range(-km+1, 0):
        sigma += torch.bitwise_xor(
            A_history[j],
            A_history[j-1]
        ).sum().item()

    sigma = sigma / ne
    return sigma == 0


# NGF
def NGF_scheme(
    A,
    P,
    maxIteration=10,
    km=3,
    device="cuda"
):
    """
    Full NGF scheme exactly matching Algorithm 2 + Eqs (1)-(5)
    """

    # ---- setup ----
    P = list(P)
    P_index = {v:i for i,v in enumerate(P)}

    d = next(iter(A["node_features"].values())).shape[-1]

    Ne  = EdgeUpdateNet(d).to(device)
    NDs = DownstreamNet(d).to(device)
    NUs = UpstreamNet(d).to(device)
    Nn  = NodeUpdateNet(d).to(device)

    h0 = {v: A["node_features"][v].clone() for v in P}
    h_prev = {v: h0[v].clone() for v in P}

    A_history = []
    k = 1

    # ---- main loop ----
    while True:

        if k >= maxIteration:
            break

        # record adjacency matrix
        A_history.append(adjacency_matrix(A, P))

        h_curr = {}

        for v in P:
            hDs_v = torch.zeros(d, device=device)
            hUs_v = torch.zeros(d, device=device)

            for u in A["adjacency"][v]:

                # Eq. (1): edge update
                e_prev = A["edge_features"][(v,u)]
                e_k = Ne(h_prev[v], h_prev[u], e_prev)
                A["edge_features"][(v,u)] = e_k

                # path-based direction
                if path_distance(u, P_index) < path_distance(v, P_index):
                    hDs_v += NDs(h0[v], h_prev[v], e_k)

                if path_distance(u, P_index) > path_distance(v, P_index):
                    hUs_v += NUs(h0[v], h_prev[v], e_k)

            # Eq. (4): node update
            h_curr[v] = Nn(hDs_v, hUs_v)
            A["node_features"][v] = h_curr[v]

        # ---- convergence check (Eq. 5) ----
        if has_converged_adj(A_history, km):
            break

        h_prev = h_curr
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

