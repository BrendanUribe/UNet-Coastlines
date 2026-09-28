# turns the model's coastline band (2 px wide) into thin lines of points
#   1. thin     - shrink the band to its 1 pixel center line (skeletonize)
#   2. refine   - nudge each point toward where the coastline probability is highest (sub-pixel)
#   3. trace    - walk along the line to get ordered x,y point lists, one per visible stretch of coast
# all coordinates here are in MODEL pixels (x = column, y = row), use to_original() to map back

import numpy as np # math
from skimage.morphology import skeletonize # thinning


def thin_band(band): # band [H, W] of 0/1 -> 1 pixel wide center line [H, W] bool
    return skeletonize(band.astype(bool))


def refine_subpixel(line, prob, band, radius=2): # each line pixel -> probability weighted center of its neighborhood
    # only band pixels count, so points get pulled toward the strongest coastline response
    H, W = prob.shape
    weights_map = prob * band
    ys, xs = np.nonzero(line)
    refined = np.zeros((len(xs), 2)) # columns: x, y
    for i, (y, x) in enumerate(zip(ys, xs)):
        y0, y1 = max(y - radius, 0), min(y + radius + 1, H)
        x0, x1 = max(x - radius, 0), min(x + radius + 1, W)
        w = weights_map[y0:y1, x0:x1]
        if w.sum() > 0:
            gy, gx = np.mgrid[y0:y1, x0:x1]
            refined[i] = [(w * gx).sum() / w.sum(), (w * gy).sum() / w.sum()]
        else:
            refined[i] = [x, y]
    return xs, ys, refined


def trace_segments(line, min_length=3): # 1 px line -> list of ordered point arrays [[x, y], ...]
    pixels = set(zip(*np.nonzero(line))) # (y, x) pairs
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbors(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in steps if (p[0] + dy, p[1] + dx) in pixels]

    visited = set()
    segments = []
    # start at line ends first (1 neighbor), then whatever is left (closed loops like islands)
    ends = [p for p in pixels if len(neighbors(p)) == 1]
    for start in ends + list(pixels):
        if start in visited:
            continue
        segment = [start]
        visited.add(start)
        current = start
        while True:
            nxt = [n for n in neighbors(current) if n not in visited]
            if not nxt:
                break
            # prefer straight (4-connected) steps so the walk doesn't skip corners
            nxt.sort(key=lambda n: abs(n[0] - current[0]) + abs(n[1] - current[1]))
            current = nxt[0]
            visited.add(current)
            segment.append(current)
        if len(segment) >= min_length:
            segments.append(np.array([[x, y] for y, x in segment], dtype=float))
    return segments


def to_original(points, scale, pad_x, pad_y): # model pixel coords [N, 2] (x, y) -> original image pixel coords
    points = np.asarray(points, dtype=float)
    return np.column_stack([(points[:, 0] - pad_x + 0.5) * scale - 0.5,
                            (points[:, 1] - pad_y + 0.5) * scale - 0.5])


# camera from case_type 3000 in functions.py: angle covered by one ORIGINAL image pixel
FOCAL_LEN_MM, PIXEL_SIZE_MM = 35, 4.96e-3
MRAD_PER_PIXEL = PIXEL_SIZE_MM / FOCAL_LEN_MM * 1000


def predicted_line(edge_band, edge_prob, scale, pad_x, pad_y): # model band -> thin line in ORIGINAL pixels
    # returns segments (list of [N, 2] arrays), sub-pixel points [N, 2], pixel-center points [N, 2]
    line = thin_band(edge_band)
    line_xs, line_ys, refined = refine_subpixel(line, edge_prob, edge_band)
    if len(refined) == 0:
        return [], np.zeros((0, 2)), np.zeros((0, 2))
    refined_at = {(x, y): r for x, y, r in zip(line_xs, line_ys, refined)} # line pixel -> refined position
    segments = [to_original([refined_at[(int(x), int(y))] for x, y in seg], scale, pad_x, pad_y)
                for seg in trace_segments(line)]
    return (segments, to_original(refined, scale, pad_x, pad_y),
            to_original(np.column_stack([line_xs, line_ys]), scale, pad_x, pad_y))


def true_boundary_points(label, water=1, land=2): # full res label map -> exact land/water boundary points [N, 2] (x, y)
    # midpoints between neighboring land and water pixels
    lw = np.isin(label, [water, land])
    pair_x = lw[:, :-1] & lw[:, 1:] & (label[:, :-1] != label[:, 1:]) # left-right land/water pairs
    pair_y = lw[:-1, :] & lw[1:, :] & (label[:-1, :] != label[1:, :]) # up-down land/water pairs
    ty, tx = np.nonzero(pair_x)
    uy, ux = np.nonzero(pair_y)
    return np.concatenate([np.column_stack([tx + 0.5, ty]), np.column_stack([ux, uy + 0.5])]).astype(float)
