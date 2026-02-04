import torch
import numpy as np
import cv2
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F

# Superpixel Extraction
def get_superpixel_pixels(labels):
    """
    Returns dict: {superpixel_id: [(y,x), ...]}
    """
    sp_dict = defaultdict(list)
    H, W = labels.shape
    for y in range(H):
        for x in range(W):
            if labels[y, x] >= 0:
                sp_dict[int(labels[y, x])].append((y, x))
    return sp_dict

# Get Superpixel Patch (V)
def get_superpixel_patch(image, pixels, pad=2):
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]

    y1, y2 = max(min(ys)-pad, 0), min(max(ys)+pad, image.shape[0])
    x1, x2 = max(min(xs)-pad, 0), min(max(xs)+pad, image.shape[1])

    patch = image[y1:y2, x1:x2]
    return patch

# Resizing the patch
def resize_patch(patch, size=32):
    return cv2.resize(patch, (size, size))

# Encoder Network (N)
class PatchEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Generate Graph (S)
def generate_graph(labels):
    H, W = labels.shape
    adj = defaultdict(set)

    for y in range(H-1):
        for x in range(W-1):
            v = labels[y, x]
            for dy, dx in [(1,0),(0,1)]:
                u = labels[y+dy, x+dx]
                if v != u and v >= 0 and u >= 0:
                    adj[v].add(u)
                    adj[u].add(v)
    return adj

# Construction of Adacency Network (A)
def construct_graph_A(image, labels, encoder, device="cuda"):
    encoder = encoder.to(device)
    encoder.eval()

    sp_pixels = get_superpixel_pixels(labels)
    adjacency = generate_graph(labels)

    A = {
        "node_features": {},
        "edge_features": {},
        "adjacency": adjacency
    }

    P = list(sp_pixels.keys())  # Path P (all nodes)

    for v in P:
        # s_p^v ← GetSuperPixelPatch(v)
        patch_v = get_superpixel_patch(image, sp_pixels[v])

        # ResizeSuperPixelPatch
        patch_v = resize_patch(patch_v)

        # h_v^0 ← N(s_p^v)
        patch_v = torch.tensor(patch_v).permute(2,0,1).float().unsqueeze(0)/255.
        h_v = encoder(patch_v.to(device)).squeeze(0)

        # UpdateNodeFeatures
        A["node_features"][v] = h_v

        # Ψ_v ← GetAdjacentNodes
        for u in adjacency[v]:
            if (v, u) in A["edge_features"]:
                continue

            # s_p^u
            patch_u = get_superpixel_patch(image, sp_pixels[u])
            patch_u = resize_patch(patch_u)

            patch_u = torch.tensor(patch_u).permute(2,0,1).float().unsqueeze(0)/255.
            h_u = encoder(patch_u.to(device)).squeeze(0)

            # e_vu ← h_v - h_u
            e_vu = h_v - h_u

            # e_vu^0 ← N_e(e_vu)
            # (identity or MLP)
            A["edge_features"][(v,u)] = e_vu
            A["edge_features"][(u,v)] = -e_vu

    return A

