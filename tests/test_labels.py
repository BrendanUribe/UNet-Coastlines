"""Regression tests for the label pipeline.

Each test here pins down one of the label defects that made the original
3-class pipeline learn the wrong thing.  They build synthetic renders that
match what the fixed POV-Ray scenes emit, so they run without POV-Ray, SPICE
or a GPU.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import labels as L  # noqa: E402


H = W = 64
CX = CY = 32.0
R = 20.0


def _disc():
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - CX) ** 2 + (yy - CY) ** 2) <= R ** 2


def _scene(land_cols=slice(20, 30), cloud_cols=None, lit_cols=slice(0, W)):
    """Build (image, land/sea mask, cloud mask) as the fixed renderer would.

    Space is the red sentinel; the mask body is black ocean / white land; the
    RGB image is black wherever the Sun does not reach.
    """
    disc = _disc()

    land = np.zeros((H, W), bool)
    land[:, land_cols] = True
    land &= disc

    cloud = np.zeros((H, W), bool)
    if cloud_cols is not None:
        cloud[:, cloud_cols] = True
        cloud &= disc

    lit = np.zeros((H, W), bool)
    lit[:, lit_cols] = True

    mask = np.zeros((H, W, 3), np.float32)
    mask[~disc] = L.SPACE_SENTINEL_RGB
    mask[disc & land] = (1.0, 1.0, 1.0)

    cloud_mask = np.zeros((H, W, 3), np.float32)
    cloud_mask[~cloud] = L.SPACE_SENTINEL_RGB

    # RGB render: ambient 0, so unlit surface is genuinely black.
    image = np.zeros((H, W, 3), np.float32)
    body = disc & lit
    image[body] = 0.45
    image[body & land] = 0.30
    image[body & cloud] = 0.95
    return image, mask, cloud_mask, disc, land, cloud, lit


def test_space_is_not_water():
    """The original bug: POV-Ray's black background == the ocean value."""
    image, mask, cloud_mask, disc, *_ = _scene()
    seg = L.compose_seg_label(image, mask, cloud_mask)
    assert (seg[~disc] == L.SPACE).all()
    assert not (seg[~disc] == L.WATER).any()


def test_night_side_is_not_water_or_cloudless():
    """Unlit land must not be labelled ocean, unlit cloud must not vanish."""
    lit = slice(0, 32)
    image, mask, cloud_mask, disc, land, cloud, lit_m = _scene(
        cloud_cols=slice(40, 50), lit_cols=lit
    )
    seg = L.compose_seg_label(image, mask, cloud_mask)

    dark_land = disc & land & ~lit_m
    dark_cloud = disc & cloud & ~lit_m
    if dark_land.any():
        assert (seg[dark_land] == L.NIGHT).all()
    assert dark_cloud.any()
    assert (seg[dark_cloud] == L.NIGHT).all()
    assert not (seg[dark_cloud] == L.CLOUD).any()


def test_terminator_is_not_a_coastline():
    """The headline bug: a Sobel edge on the shaded mask made the terminator a
    coastline label, which is what the network then drew across the disc."""
    image, mask, cloud_mask, *_ = _scene(land_cols=slice(2, 10), lit_cols=slice(0, 32))
    seg = L.compose_seg_label(image, mask, cloud_mask)
    edges, valid = L.make_edge_targets(seg)

    term = edges[L.EDGE_TERMINATOR] > 0
    assert term.any(), "expected a terminator in this scene"
    # No terminator pixel may be supervised as coastline.
    supervised_coast = (edges[L.EDGE_COASTLINE] > 0) & (valid[L.EDGE_COASTLINE] > 0)
    assert not (term & supervised_coast).any()


def test_limb_excludes_the_dark_limb():
    """Only the illuminated limb is a target; the dark limb is masked out so a
    conic fit is not biased by unobservable horizon points."""
    image, mask, cloud_mask, *_ = _scene(lit_cols=slice(0, 32))
    seg = L.compose_seg_label(image, mask, cloud_mask)
    edges, valid = L.make_edge_targets(seg)

    dark_limb = L.boundary_between(seg, [L.SPACE], [L.NIGHT])
    assert dark_limb.any()
    assert (valid[L.EDGE_LIMB][dark_limb] == 0).all()
    assert (edges[L.EDGE_LIMB][dark_limb] == 0).all()

    # Away from the junction, the illuminated limb is fully supervised.  The
    # handful of pixels where the terminator meets the limb touch both lit
    # surface and night, so they are legitimately ambiguous and masked out.
    lit_limb = L.boundary_between(seg, [L.SPACE], list(L.LIT_EARTH))
    junction = lit_limb & dark_limb
    assert lit_limb.any()
    assert junction.sum() <= 4, "junction should be a couple of pixels, not a region"
    assert (valid[L.EDGE_LIMB][lit_limb & ~junction] == 1).all()
    assert (edges[L.EDGE_LIMB][lit_limb & ~junction] == 1).all()


def test_coastline_unsupervised_under_cloud_and_at_night():
    image, mask, cloud_mask, disc, land, cloud, lit_m = _scene(
        land_cols=slice(20, 44), cloud_cols=slice(24, 30), lit_cols=slice(0, 40)
    )
    seg = L.compose_seg_label(image, mask, cloud_mask)
    edges, valid = L.make_edge_targets(seg)

    assert (valid[L.EDGE_COASTLINE][seg == L.CLOUD] == 0).all()
    assert (valid[L.EDGE_COASTLINE][seg == L.NIGHT] == 0).all()
    assert (valid[L.EDGE_COASTLINE][seg == L.SPACE] == 0).all()
    assert (valid[L.EDGE_COASTLINE][seg == L.LAND] == 1).all()
    assert (valid[L.EDGE_COASTLINE][seg == L.WATER] == 1).all()


def test_real_coastline_is_still_found():
    """Sanity: the fix must not suppress the signal we actually want."""
    image, mask, cloud_mask, disc, land, *_ = _scene(land_cols=slice(20, 30))
    seg = L.compose_seg_label(image, mask, cloud_mask)
    edges, valid = L.make_edge_targets(seg)

    coast = (edges[L.EDGE_COASTLINE] > 0) & (valid[L.EDGE_COASTLINE] > 0)
    assert coast.sum() > 10
    # and every coastline pixel really does sit on a land/water interface
    assert np.isin(seg[coast], [L.LAND, L.WATER]).all()


def test_legacy_masks_are_detected():
    """Masks rendered before the fix have no sentinel and must be refused."""
    image, mask, cloud_mask, *_ = _scene()
    assert L.has_space_sentinel(mask)

    legacy = mask.copy()
    legacy[L.is_space_sentinel(mask)] = 0.0  # old renderer: space was black
    assert not L.has_space_sentinel(legacy)


def test_class_coverage_is_exhaustive():
    image, mask, cloud_mask, *_ = _scene(
        cloud_cols=slice(34, 40), lit_cols=slice(0, 40)
    )
    seg = L.compose_seg_label(image, mask, cloud_mask)
    assert set(np.unique(seg)) <= set(L.SEG_CLASS_NAMES)
    assert (seg < L.NUM_SEG_CLASSES).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_antialiased_limb_is_classified_by_majority_coverage():
    """POV-Ray renders the masks with Antialias=On, Threshold=0.0, so limb
    pixels are a blend of the red space sentinel and the surface beneath.

    The sentinel is chosen so this stays decodable: for a pixel that is
    fraction ``a`` space over ocean (0,0,0) the result is ``(a,0,0)``, and over
    land (1,1,1) it is ``(1,1-a,1-a)``. In both cases ``R - G == a`` exactly,
    so thresholding at 0.5 assigns the pixel to whichever actually covers most
    of it. Without that property, antialiased limb pixels over land would
    decode as water and manufacture a false coastline all along the limb.
    """
    def quantise(v):
        return np.round(np.clip(v, 0, 1) * 255) / 255.0

    lit = np.full((1, 1, 3), 0.5, np.float32)
    no_cloud = np.array(L.SPACE_SENTINEL_RGB, np.float32).reshape(1, 1, 3)

    for surface, expected_surface in (((0, 0, 0), L.WATER), ((1, 1, 1), L.LAND)):
        for a in (0.0, 0.1, 0.25, 0.4, 0.45, 0.49):
            px = quantise(a * np.array(L.SPACE_SENTINEL_RGB) + (1 - a) * np.array(surface))
            seg = L.compose_seg_label(lit, px.reshape(1, 1, 3).astype(np.float32), no_cloud)
            assert seg[0, 0] == expected_surface, f"a={a} surface={surface} -> {seg[0,0]}"

        for a in (0.55, 0.6, 0.75, 0.9, 1.0):
            px = quantise(a * np.array(L.SPACE_SENTINEL_RGB) + (1 - a) * np.array(surface))
            seg = L.compose_seg_label(lit, px.reshape(1, 1, 3).astype(np.float32), no_cloud)
            assert seg[0, 0] == L.SPACE, f"a={a} surface={surface} -> {seg[0,0]}"


def test_antialiased_cloud_edge_is_majority_coverage():
    """Same property for the cloud mask: white cloud over the red sentinel."""
    def quantise(v):
        return np.round(np.clip(v, 0, 1) * 255) / 255.0

    lit = np.full((1, 1, 3), 0.5, np.float32)
    ocean = np.zeros((1, 1, 3), np.float32)

    for a, expected in ((0.0, L.WATER), (0.4, L.WATER), (0.6, L.CLOUD), (1.0, L.CLOUD)):
        px = quantise(a * np.array([1.0, 1.0, 1.0]) + (1 - a) * np.array(L.SPACE_SENTINEL_RGB))
        seg = L.compose_seg_label(lit, ocean, px.reshape(1, 1, 3).astype(np.float32))
        assert seg[0, 0] == expected, f"cloud coverage {a} -> {L.SEG_CLASS_NAMES[seg[0,0]]}"
