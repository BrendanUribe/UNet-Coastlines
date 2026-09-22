"""Rank frames by how strong a terminator they contain.

For picking a demonstration frame: the terminator channel is only meaningful
where a frame has both a lit side and a night side, and a full-phase frame
shows nothing. Prints the frames with the most terminator pixels.

    python src/find_terminator.py
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, 'src')
import labels as L

IM, MK, CM = 'dataset/images', 'dataset/masks', 'dataset/cloud masks'
if len(sys.argv) > 1: IM, MK, CM = sys.argv[1], sys.argv[2], sys.argv[3]

def rd(p):
    a = cv2.imread(p, cv2.IMREAD_COLOR)
    return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32)/255

rows = []
for f in sorted(os.listdir(IM)):
    if not f.endswith('.png') or 'MASK' in f: continue
    i = f[len('earth_img_'):-4]
    img, m, c = rd(os.path.join(IM,f)), rd(os.path.join(MK,f'earth_img_MASK{i}.png')), rd(os.path.join(CM,f'earth_img_CLOUDMASK{i}.png'))
    if img is None or m is None or c is None: continue
    if img.shape[:2] != m.shape[:2] or img.shape[:2] != c.shape[:2]: continue
    seg = L.compose_seg_label(img, m, c)
    e, _ = L.make_edge_targets(seg)
    night = (seg == L.NIGHT).mean()
    lit = np.isin(seg, list(L.LIT_EARTH)).mean()
    term = int((e[L.EDGE_TERMINATOR] > 0).sum())
    coast = int((e[L.EDGE_COASTLINE] > 0).sum())
    if night > 0.01 and lit > 0.05:
        rows.append((term, f, night, lit, coast))

rows.sort(reverse=True)
print(f"{len(rows)} frames with both a lit side and a night side\n")
print(f"{'frame':>22}{'term px':>9}{'night%':>8}{'lit%':>7}{'coast px':>10}")
for term, f, n, l, co in rows[:10]:
    print(f"{f:>22}{term:>9}{n*100:>7.1f}{l*100:>7.1f}{co:>10}")
if not rows:
    print("NONE - every frame is full-phase or fully dark.")
    print("The terminator channel has nothing to learn from; render higher phase angles.")
