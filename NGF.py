import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
class EdgeUpdateNet(nn.Module):
    # Eq. (1)
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, h_v, h_u, e_vu):
        return self.net(torch.cat([h_v, h_u, e_vu], dim=-1))
class DownstreamNet(nn.Module):
    # Eq. (3)
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(3 * d, d)

    def forward(self, h0_v, h_prev_v, e_vu):
        return self.fc(torch.cat([h0_v, h_prev_v, e_vu], dim=-1))

class UpstreamNet(nn.Module):
    # Eq. (2)
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(3 * d, d)

    def forward(self, h0_v, h_prev_v, e_vu):
        return self.fc(torch.cat([h0_v, h_prev_v, e_vu], dim=-1))
class NodeUpdateNet(nn.Module):
    # Eq. (4)
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, h_ds, h_us):
        return self.net(torch.cat([h_ds, h_us], dim=-1))

def NGF_scheme(
    A,
    P,
    maxIteration=6,
    device="cuda"
):
    """
    A:
      A["adjacency"]     : dict {v: set(u)}
      A["node_features"]: dict {v: h_v^0}
      A["edge_features"]: dict {(v,u): e_vu^0}

    P: traversal order (list of nodes)
    """

    P = list(P)
    P_index = {v: i for i, v in enumerate(P)}

    d = next(iter(A["node_features"].values())).shape[-1]

    Ne  = EdgeUpdateNet(d).to(device)
    NDs = DownstreamNet(d).to(device)
    NUs = UpstreamNet(d).to(device)
    Nn  = NodeUpdateNet(d).to(device)

    # Initial embeddings
    h0 = {v: A["node_features"][v].clone() for v in P}
    h_prev = {v: h0[v].clone() for v in P}

    for k in range(1, maxIteration + 1):
        h_curr = {}

        for v in P:
            hDs_v = torch.zeros(d, device=device)
            hUs_v = torch.zeros(d, device=device)

            for u in A["adjacency"][v]:
                # ---- Eq. (1): edge update ----
                e_prev = A["edge_features"][(v, u)]
                e_k = Ne(h_prev[v], h_prev[u], e_prev)
                A["edge_features"][(v, u)] = e_k

                # ---- Eq. (2–3): direction via path ----
                if P_index[u] < P_index[v]:
                    hDs_v += NDs(h0[v], h_prev[v], e_k)
                elif P_index[u] > P_index[v]:
                    hUs_v += NUs(h0[v], h_prev[v], e_k)

            # ---- Eq. (4): node update ----
            h_curr[v] = Nn(hDs_v, hUs_v)
            A["node_features"][v] = h_curr[v]

        h_prev = h_curr

    return A

def visualize_NGF(
    image_path,
    labels,
    A,
    savename=None
):
    image = cv2.imread(image_path)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # ---- compute superpixel centroids ----
    centroids = {}
    for sp in np.unique(labels):
        if sp < 0:
            continue
        ys, xs = np.where(labels == sp)
        centroids[int(sp)] = (int(xs.mean()), int(ys.mean()))

    vis = image.copy()

    # ---- compute edge strengths ----
    edges = []
    for (v, u), e in A["edge_features"].items():
        if v in centroids and u in centroids:
            w = torch.norm(e).item()
            edges.append((v, u, w))

    w_max = max(w for _, _, w in edges) + 1e-6

    # ---- draw edges (weighted) ----
    for v, u, w in edges:
        alpha = w / w_max

        thickness = int(1 + 4 * alpha)
        color = (
            0,
            int(255 * alpha),
            int(255 * (1 - alpha))
        )

        cv2.line(
            vis,
            centroids[v],
            centroids[u],
            color=color,
            thickness=thickness
        )

    # ---- draw nodes ----
    for x, y in centroids.values():
        cv2.circle(vis, (x, y), 2, (255, 255, 0), -1)

    if savename:
        cv2.imwrite(savename, vis)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Neural Graph Featurization (NGF)")
    plt.axis("off")
    plt.show()

    return vis
