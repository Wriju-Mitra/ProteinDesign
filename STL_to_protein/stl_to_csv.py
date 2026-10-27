#!/usr/bin/env python3
"""
stl_to_csv.py - turn an STL mesh into an x,y,z point-cloud CSV that can be
loaded the same way as Shape.csv (no header, 3 columns, one point per row).

Needs only numpy.  Reads binary and ASCII STL.

Examples
--------
    python stl_to_csv.py part.stl Shape.csv
    python stl_to_csv.py part.stl Shape.csv --pitch 0.04 --longest 1.0
    python stl_to_csv.py hollow_thing.stl Shape.csv --mode surface

--mode solid    (default) fills the inside of a closed mesh with a regular
                grid of points, like the example Shape.csv.
--mode surface  only puts points on the skin of the mesh (works even if the
                mesh has holes / is not watertight).

The mesh is rescaled so its longest side equals --longest (default 1.0, same
as the example) and centred on the origin. --pitch is the grid spacing in the
rescaled units (example Shape.csv uses 0.05, i.e. ~20 points along the
longest side).
"""
import argparse
import struct
import sys

import numpy as np


# --------------------------------------------------------------------------
# STL loading
# --------------------------------------------------------------------------
def load_stl(path):
    """Return triangles as an (N, 3, 3) float array: [triangle, vertex, xyz]."""
    with open(path, "rb") as f:
        data = f.read()

    # Binary STL: 80-byte header, uint32 count, then 50 bytes per triangle.
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n and n > 0:
            dt = np.dtype([("normal", "<f4", (3,)),
                           ("v", "<f4", (3, 3)),
                           ("attr", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(np.float64)

    # ASCII STL
    verts = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex":
            verts.append([float(p) for p in parts[1:4]])
    if not verts or len(verts) % 3:
        raise ValueError(f"{path} does not look like a valid STL file")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def is_watertight(tris, decimals=6):
    """True if every edge is shared by exactly two triangles."""
    flat = np.round(tris.reshape(-1, 3), decimals)
    _, ids = np.unique(flat, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    edges = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


# --------------------------------------------------------------------------
# Solid fill: cast a ray along +z through every (x, y) grid column, find where
# it crosses the surface, and mark grid points between entry/exit as inside.
# --------------------------------------------------------------------------
def solid_points(tris, pitch):
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    # Grid points are cell centres. A tiny offset keeps rays from landing
    # exactly on triangle edges (which would double-count crossings).
    jitter = np.array([1.234e-5, 2.718e-5, 0.0]) * pitch
    xs = lo[0] + (np.arange(n[0]) + 0.5) * pitch + jitter[0]
    ys = lo[1] + (np.arange(n[1]) + 0.5) * pitch + jitter[1]
    nx, ny, nz = n

    cols, zs = [], []
    for a, b, c in tris:
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-15:          # triangle is vertical, ray never crosses it
            continue
        i0 = max(int(np.floor((min(a[0], b[0], c[0]) - lo[0]) / pitch - 0.5)), 0)
        i1 = min(int(np.ceil((max(a[0], b[0], c[0]) - lo[0]) / pitch - 0.5)), nx - 1)
        j0 = max(int(np.floor((min(a[1], b[1], c[1]) - lo[1]) / pitch - 0.5)), 0)
        j1 = min(int(np.ceil((max(a[1], b[1], c[1]) - lo[1]) / pitch - 0.5)), ny - 1)
        if i1 < i0 or j1 < j0:
            continue
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px, py = xs[I], ys[J]
        l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
        l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if hit.any():
            cols.append((I[hit] * ny + J[hit]))
            zs.append(l1[hit] * a[2] + l2[hit] * b[2] + l3[hit] * c[2])

    if not cols:
        return np.zeros((0, 3))
    cols = np.concatenate(cols)
    zs = np.concatenate(zs)

    # Each crossing flips inside/outside for every grid layer above it.
    k_start = np.clip(np.ceil((zs - lo[2]) / pitch - 0.5).astype(int), 0, nz)
    toggles = np.zeros((nx * ny, nz + 1), dtype=np.int32)
    np.add.at(toggles, (cols, k_start), 1)
    inside = (np.cumsum(toggles, axis=1) % 2)[:, :nz].astype(bool)

    col_idx, k_idx = np.nonzero(inside)
    zc = lo[2] + (k_idx + 0.5) * pitch
    # report the clean grid positions (undo the tiny ray offset)
    return np.column_stack([xs[col_idx // ny] - jitter[0], ys[col_idx % ny] - jitter[1], zc])


# --------------------------------------------------------------------------
# Surface: sprinkle samples over every triangle, snap to the grid, de-duplicate.
# --------------------------------------------------------------------------
def surface_points(tris, pitch, seed=0):
    rng = np.random.default_rng(seed)
    lo = tris.reshape(-1, 3).min(axis=0)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    counts = np.maximum(np.ceil(area / (0.25 * pitch ** 2)).astype(int), 1)

    tri_idx = np.repeat(np.arange(len(tris)), counts)
    r1, r2 = rng.random(len(tri_idx)), rng.random(len(tri_idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    a, b, c = tris[tri_idx, 0], tris[tri_idx, 1], tris[tri_idx, 2]
    pts = a + r1[:, None] * (b - a) + r2[:, None] * (c - a)

    cells = np.unique(np.floor((pts - lo) / pitch).astype(int), axis=0)
    return lo + (cells + 0.5) * pitch


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl")
    ap.add_argument("csv")
    ap.add_argument("--pitch", type=float, default=0.05, help="grid spacing after rescaling (default 0.05)")
    ap.add_argument("--longest", type=float, default=1.0,
                    help="rescale so the longest bounding-box side has this length (default 1.0)")
    ap.add_argument("--mode", choices=["solid", "surface"], default="solid")
    args = ap.parse_args()

    tris = load_stl(args.stl)
    print(f"Loaded {len(tris)} triangles")

    size = tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)
    if args.longest > 0:
        tris = tris * (args.longest / size.max())
    centre = (tris.reshape(-1, 3).max(0) + tris.reshape(-1, 3).min(0)) / 2
    tris = tris - centre

    if args.mode == "solid":
        if not is_watertight(tris):
            print("WARNING: mesh is not watertight; solid fill may have gaps or stripes. "
                  "Repair the mesh or use --mode surface.", file=sys.stderr)
        pts = solid_points(tris, args.pitch)
    else:
        pts = surface_points(tris, args.pitch)

    if len(pts) == 0:
        sys.exit("No points produced - check the mesh and --pitch.")

    np.savetxt(args.csv, pts, delimiter=",", fmt="%.6g")   # no header, like Shape.csv
    ext = pts.max(0) - pts.min(0)
    print(f"Wrote {len(pts)} points to {args.csv}  (extent x={ext[0]:.3f} y={ext[1]:.3f} z={ext[2]:.3f})")


if __name__ == "__main__":
    main()
