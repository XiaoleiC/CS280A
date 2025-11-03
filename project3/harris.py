
import numpy as np
from skimage.feature import corner_harris, peak_local_max


def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def anms(coords, h, num_keep=500, c_robust=0.9):
    """
    Adaptive Non-Maximal Suppression (ANMS) as described in
    Brown et al., "Multi-Image Matching using Multi-Scale Oriented Patches".

    Args:
        coords: np.ndarray shaped (2, N) with integer (ys, xs) corner coordinates.
        h:      2D np.ndarray Harris response map (same HxW as image).
        num_keep: number of interest points to retain.
        c_robust: robust factor; j must be stronger than i by this factor (default 0.9).

    Returns:
        coords_keep: np.ndarray (2, K) with K=min(num_keep, N) selected coordinates.
        radii:       np.ndarray (K,) suppression radii of the kept points.
    """
    if coords.size == 0:
        return coords, np.array([])

    ys, xs = coords
    vals = h[ys, xs].astype(float)
    N = vals.shape[0]

    # Sort by strength descending
    order = np.argsort(-vals)
    ys_s, xs_s, vals_s = ys[order], xs[order], vals[order]

    # For each point i, find min distance to any *stronger* point j with f(x_j) > c * f(x_i).
    # The strongest point gets inf radius (by definition it has no stronger neighbor).
    radii = np.full(N, np.inf, dtype=float)

    # Brute-force; OK for typical N (a few thousand). Vectorize per i over all stronger j.
    for i in range(1, N):
        hi = vals_s[i]
        # indices of stronger points meeting robust constraint
        good = np.where(vals_s[:i] > c_robust * hi)[0]
        if good.size == 0:
            # No qualifying stronger neighbor; leave inf (will rank high)
            continue
        dy = ys_s[i] - ys_s[good]
        dx = xs_s[i] - xs_s[good]
        dist2 = dx * dx + dy * dy
        # store on the original index slot
        radii[order[i]] = np.min(dist2)

    # Replace inf with large finite for sorting; ensures the global max is kept
    if np.isfinite(radii).any():
        max_finite = np.max(radii[np.isfinite(radii)])
    else:
        max_finite = 0.0
    radii[np.isinf(radii)] = max_finite + 1.0

    # Keep top-k by radius
    k = min(num_keep, N)
    keep_idx = np.argsort(-radii)[:k]

    # Return in (ys, xs) with corresponding radii
    return np.vstack([ys[keep_idx], xs[keep_idx]]), radii[keep_idx]