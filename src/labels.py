"""Single source of truth for label semantics.

Everything that decides "what is this pixel" lives here, so the renderer
(src/functions.py), the dataset, the loss and the evaluation can never drift
apart.

Why the label scheme changed
----------------------------
The original pipeline used 3 classes (water / land / cloud) derived from a
*shaded* POV-Ray render of the land-sea mask.  That produced three first-order
label defects:

  1. POV-Ray's default background is black, which is the same value the mask
     uses for ocean, so empty space was labelled as water.
  2. The mask was rendered with ``ambient 0 / diffuse 1``, i.e. lit by the Sun.
     Past the terminator, land rendered black and was labelled as water; clouds
     likewise vanished.  Because ``brilliance 0`` makes the falloff a step, the
     terminator became a hard edge in the mask and therefore a *coastline*
     label once the Sobel edge target was computed.
  3. The cloud sphere was clipped to the Sun-facing hemisphere, which is not
     the camera-facing hemisphere at high phase angle.

The renderer now emits pure geometry (``ambient 1 / diffuse 0``) on a sentinel
background, so the masks describe *where things are* independent of lighting.

That fix has a direct consequence: the geometric mask now also describes the
night side, which is not observable in visible light.  Training a network to
predict land/water there would be asking it to hallucinate.  So illumination
comes back in as its own class, recovered from the RGB image rather than baked
into the mask.  This separation is what lets us emit the three boundary types
optical navigation actually needs:

  * COASTLINE  - landmark matching
  * LIMB       - the illuminated horizon; the conic that is fitted for a
                 position fix (Christian & Robinson, JGCD 39(12), 2016)
  * TERMINATOR - must be *excluded* from a limb fit, so it has to be detected
                 rather than ignored (Christian, IEEE Access 9, 2021;
                 Mortari et al., JSR 53(3), 2016)
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Segmentation classes
# --------------------------------------------------------------------------
SPACE = 0
WATER = 1
LAND = 2
CLOUD = 3
NIGHT = 4

NUM_SEG_CLASSES = 5

SEG_CLASS_NAMES = {
    SPACE: "space",
    WATER: "water",
    LAND: "land",
    CLOUD: "cloud",
    NIGHT: "night",
}

#: Classes that make up the illuminated Earth disc.
LIT_EARTH = (WATER, LAND, CLOUD)

#: Classes that make up the Earth disc regardless of illumination.
EARTH_DISC = (WATER, LAND, CLOUD, NIGHT)

# --------------------------------------------------------------------------
# Edge channels
# --------------------------------------------------------------------------
EDGE_COASTLINE = 0
EDGE_LIMB = 1
EDGE_TERMINATOR = 2

NUM_EDGE_CHANNELS = 3

EDGE_CHANNEL_NAMES = {
    EDGE_COASTLINE: "coastline",
    EDGE_LIMB: "limb",
    EDGE_TERMINATOR: "terminator",
}

# --------------------------------------------------------------------------
# Renderer contract
# --------------------------------------------------------------------------
#: Background colour written by gen_moon_earthMASK / gen_moon_earthCLOUDMASK.
#: Pure red is unambiguous against the greyscale mask body: space has R >> G,
#: ocean is (0,0,0) and land is (1,1,1).  MUST match the ``background { }``
#: statement in src/functions.py.
SPACE_SENTINEL_RGB = (1.0, 0.0, 0.0)

#: A pixel is space when (R - G) exceeds this.
SENTINEL_MIN_CHROMA = 0.5

#: Rec.709 luma below this counts as unilluminated.  The RGB renders use
#: ``ambient 0.0``, so the night side is genuinely black rather than merely
#: dark; the threshold only has to clear sensor/JPEG noise.
DEFAULT_NIGHT_LUMA = 0.03

_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of a float ``(H, W, 3)`` image in [0, 1]."""
    return rgb.astype(np.float32) @ _REC709


def is_space_sentinel(mask_rgb: np.ndarray) -> np.ndarray:
    """Boolean map of the renderer's space sentinel in a float RGB mask."""
    return (mask_rgb[..., 0] - mask_rgb[..., 1]) > SENTINEL_MIN_CHROMA


def has_space_sentinel(mask_rgb: np.ndarray, min_fraction: float = 1e-4) -> bool:
    """Whether this one mask contains any of the space sentinel.

    Absence does NOT by itself mean a pre-fix render - see
    ``is_post_fix_render``.
    """
    return bool(is_space_sentinel(mask_rgb).mean() >= min_fraction)


def is_post_fix_render(mask_rgb: np.ndarray, cloud_rgb: np.ndarray,
                       min_fraction: float = 1e-4) -> bool:
    """Whether a mask pair came from the renderer with the label fix applied.

    The land/sea mask alone is not sufficient. At close range the Earth
    overfills the sensor - below roughly 59,000 km for this camera - so a
    perfectly good post-fix mask can contain no space at all, and testing it
    alone would reject every close-range frame in the set.

    The cloud mask settles it. Its scene contains only the cloud shell, with
    non-cloud left transparent, so the sentinel background shows through
    wherever there is no cloud - which is true at any zoom level. A pre-fix
    cloud mask has a black background and no sentinel anywhere.

    So a render is post-fix if *either* mask shows the sentinel. The one case
    this cannot distinguish is a close-range frame with total cloud cover and
    no visible space, which leaves nothing for the sentinel to show through.
    """
    return (has_space_sentinel(mask_rgb, min_fraction)
            or has_space_sentinel(cloud_rgb, min_fraction))


def compose_seg_label(
    image_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    cloud_rgb: np.ndarray,
    night_luma: float = DEFAULT_NIGHT_LUMA,
) -> np.ndarray:
    """Build the 5-class segmentation label.

    Parameters
    ----------
    image_rgb, mask_rgb, cloud_rgb
        Float ``(H, W, 3)`` arrays in [0, 1], all the same size.  ``mask_rgb``
        is the land/sea render, ``cloud_rgb`` the cloud render.
    night_luma
        Luma below which an on-disc pixel counts as unilluminated.

    Returns
    -------
    ``(H, W)`` uint8 array of class ids.

    Precedence is space > night > cloud > land/water: a pixel you cannot see
    is never given a surface class.
    """
    space_raw = is_space_sentinel(mask_rgb)
    cloud = ~is_space_sentinel(cloud_rgb)

    # The cloud shell sits marginally outside the solid Earth (3.5003 vs 3.5),
    # so the visible disc is the union of the two.
    disc = (~space_raw) | cloud
    lit = luma(image_rgb) >= night_luma

    # Land is white, ocean black, in the greyscale body of the land/sea mask.
    land_tex = mask_rgb[..., 1] > 0.5

    seg = np.full(disc.shape, SPACE, dtype=np.uint8)
    seg[disc & ~lit] = NIGHT
    seg[disc & lit & cloud] = CLOUD
    seg[disc & lit & ~cloud & land_tex] = LAND
    seg[disc & lit & ~cloud & ~land_tex] = WATER
    return seg


def _shift_eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4-neighbour adjacency: pixels of ``a`` touching a pixel of ``b``."""
    out = np.zeros_like(a, dtype=bool)
    out[:-1, :] |= a[:-1, :] & b[1:, :]
    out[1:, :] |= a[1:, :] & b[:-1, :]
    out[:, :-1] |= a[:, :-1] & b[:, 1:]
    out[:, 1:] |= a[:, 1:] & b[:, :-1]
    return out


def boundary_between(seg: np.ndarray, set_a, set_b) -> np.ndarray:
    """Symmetric 4-connected boundary between two groups of classes."""
    a = np.isin(seg, np.asarray(set_a, dtype=seg.dtype))
    b = np.isin(seg, np.asarray(set_b, dtype=seg.dtype))
    return _shift_eq(a, b) | _shift_eq(b, a)


def make_edge_targets(seg: np.ndarray):
    """Derive the three boundary targets and their validity masks.

    Returns
    -------
    edges : ``(3, H, W)`` float32, 1.0 on a boundary
    valid : ``(3, H, W)`` float32, 1.0 where the channel is supervised

    A channel is left unsupervised where the boundary is real geometry but
    carries no signal in the image:

    * COASTLINE  - unsupervised off the lit, cloud-free surface.  A coast under
      cloud or past the terminator is not visible, so neither presence nor
      absence should be penalised.
    * LIMB       - only the *illuminated* limb is a target.  The space/night
      boundary is the same geometric horizon but is radiometrically invisible,
      so it is masked out instead of being labelled either way.  This matters
      downstream: a conic fitted through terminator or dark-limb points is
      biased (Christian 2021, §V).
    * TERMINATOR - always supervised; it is a high-contrast feature.
    """
    h, w = seg.shape
    edges = np.zeros((NUM_EDGE_CHANNELS, h, w), dtype=np.float32)
    valid = np.ones((NUM_EDGE_CHANNELS, h, w), dtype=np.float32)

    edges[EDGE_COASTLINE] = boundary_between(seg, [LAND], [WATER])
    edges[EDGE_LIMB] = boundary_between(seg, [SPACE], list(LIT_EARTH))
    edges[EDGE_TERMINATOR] = boundary_between(seg, [NIGHT], list(LIT_EARTH))

    observable_surface = np.isin(seg, np.array([LAND, WATER], dtype=seg.dtype))
    valid[EDGE_COASTLINE] = observable_surface.astype(np.float32)

    dark_limb = boundary_between(seg, [SPACE], [NIGHT])
    valid[EDGE_LIMB] = (~dark_limb).astype(np.float32)

    # Where a channel is unsupervised its target is meaningless, so zero it.
    # This matters at the limb/terminator junction, where a space pixel can be
    # 4-adjacent to both lit surface and night: the label there is genuinely
    # ambiguous, so it is masked out rather than resolved arbitrarily.
    edges *= valid
    return edges, valid
