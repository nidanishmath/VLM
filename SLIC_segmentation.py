import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# Mask Images
def get_fundus_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Retina is brighter than background
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# mixed distance
def mixed_distance_torch(C, P, S, m):
    """
    C : (K, 5)  [L, A, B, Y, X]
    P : (H, W, 5)
    """
    dc = (P[..., :3] - C[:, None, None, :3]).pow(2).sum(dim=-1)
    ds = (P[..., 3:] - C[:, None, None, 3:]).pow(2).sum(dim=-1)
    return torch.sqrt(dc + (m * m / (S * S)) * ds)


# segmentation
def SLIC_GPU(
    filename,
    k=400,
    m=10,
    threshold=0.05,
    max_iters=10,
    device="cuda"
):
    assert torch.cuda.is_available(), "CUDA not available"

    # --- Load image ---
    img = cv2.imread(filename)
    H, W, _ = img.shape

    # --- Fundus mask ---
    mask_np = get_fundus_mask(img)
    mask = torch.tensor(mask_np > 0, device=device)

    # --- Use GREEN channel only (best for retina) ---
    g = img[:, :, 1]
    img_green = cv2.merge([g, g, g])
    img_lab = cv2.cvtColor(img_green, cv2.COLOR_BGR2Lab)

    img_lab = torch.tensor(img_lab, dtype=torch.float32, device=device)

    # --- SLIC parameters ---
    N = mask.sum().item()
    S = int(math.sqrt(N / k))

    # --- Initialize centers INSIDE retina ---
    ys = torch.arange(S // 2, H, S, device=device)
    xs = torch.arange(S // 2, W, S, device=device)

    centers = []
    for y in ys:
        for x in xs:
            if mask[int(y), int(x)]:
                l, a, b = img_lab[int(y), int(x)]
                centers.append(torch.tensor([l, a, b, y, x], device=device))

    Ck = torch.stack(centers)
    K = Ck.shape[0]

    # --- Pixel grid ---
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    P = torch.stack([
        img_lab[..., 0],
        img_lab[..., 1],
        img_lab[..., 2],
        yy,
        xx
    ], dim=-1)

    labels = torch.full((H, W), -1, dtype=torch.long, device=device)

    prev_error = 1e20

    for it in range(max_iters):
        # --- Distance ---
        D = mixed_distance_torch(Ck, P, S, m)

        # Mask background
        D[:, ~mask] = 1e9

        labels = torch.argmin(D, dim=0)
        labels[~mask] = -1

        # --- Update centers ---
        error = 0.0
        for k_i in range(K):
            region = labels == k_i
            if region.sum() == 0:
                continue
            new_C = P[region].mean(dim=0)
            error += torch.norm(Ck[k_i] - new_C)
            Ck[k_i] = new_C

        improvement = abs(prev_error - error.item()) / prev_error
        prev_error = error.item()

        print(f"Iter {it} | Error: {error.item():.3f}")

        if improvement <= threshold:
            break

    return labels, mask

# boundary visualization

def show_segmentation(
    filename,
    savename,
    labels,
    mask,
    color=(0, 0, 255)
):
    img = cv2.imread(filename)

    # Safe conversion (GPU / CPU)
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

    H, W = labels.shape

    # Draw boundaries
    for y in range(1, H):
        valid = (labels[y] != labels[y-1]) & mask[y]
        img[y][valid] = color

    for x in range(1, W):
        valid = (labels[:, x] != labels[:, x-1]) & mask[:, x]
        img[:, x][valid] = color

    # Save output
    cv2.imwrite(savename, img)

    # ---- SHOW INLINE ----
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("SLIC Segmentation Output")
    plt.axis("off")
    plt.show()

    return img
