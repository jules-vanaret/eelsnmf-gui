import numpy as np
import pytest

from eelsnmf_gui._preprocess import (
    build_mask_result,
    build_masks,
    mask_flat_to_images,
    mask_result_from_user_masks,
    threshold_mask,
)
from eelsnmf_gui._results import MaskParams
from tests.fakes import make_si


def test_threshold_mask_uniform_image_keeps_everything():
    """A uniform image defeats multi-otsu, so the fallback keeps all pixels."""
    img = np.full((8, 8), 3.0)
    mask = threshold_mask(img, dilation_iters=0)
    assert mask.shape == (8, 8)
    assert mask.all()


def test_threshold_mask_selects_bright_region():
    """The mask keeps the bright plateau and excludes the dark background."""
    img = np.zeros((14, 14))
    img[2:10, 2:10] = 2.0
    img[4:8, 4:8] = 5.0
    mask = threshold_mask(img, dilation_iters=0)
    # The plateau (both levels) is kept, the dark border is not.
    assert mask[3:9, 3:9].all()
    assert not mask[0, :].any()


def test_threshold_mask_dilation_grows():
    """Dilation strictly enlarges the mask while keeping every base pixel."""
    img = np.zeros((16, 16))
    img[2:10, 2:10] = 2.0
    img[4:8, 4:8] = 5.0
    base = threshold_mask(img, dilation_iters=0)
    grown = threshold_mask(img, dilation_iters=2)
    assert grown.sum() > base.sum()
    assert grown[base].all()


def test_build_masks_shapes_and_flat():
    """build_masks returns per-SI masks, shapes, and a consistent flat mask."""
    datas = [np.ones((4, 5, 6)), np.ones((3, 7, 6))]
    per_si, flat, counts, shapes = build_masks(
        datas, dilation_iters=0, mode="mean"
    )
    assert [m.shape for m in per_si] == [(4, 5), (3, 7)]
    assert shapes == [(4, 5), (3, 7)]
    assert flat.shape == (4 * 5 + 3 * 7,)
    assert sum(counts) == int(flat.sum())


def test_mask_flat_to_images_roundtrip():
    """A flattened mask reshapes back into the original 2D masks."""
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:3, 2:6] = True
    images, counts = mask_flat_to_images(mask.ravel(), [(6, 8)])
    assert len(images) == 1
    np.testing.assert_array_equal(images[0], mask)
    assert counts == [int(mask.sum())]


def test_mask_flat_to_images_size_mismatch_raises():
    """A flat mask too short for the shapes raises ValueError."""
    with pytest.raises(ValueError):
        mask_flat_to_images(np.ones(10, dtype=bool), [(6, 8)])


def test_build_mask_result_from_sis():
    """build_mask_result packages per-SI masks with the requested params."""
    sis = [make_si("runA", seed=0), make_si("runB", seed=1)]
    params = MaskParams(enabled=True, dilation_iters=1, mode="mean")
    result = build_mask_result(sis, params)
    assert result.params == params
    assert len(result.per_si) == 2
    assert [m.shape for m in result.per_si] == [(4, 5), (4, 5)]
    assert result.flat.shape == (40,)
    assert all(c > 0 for c in result.counts)


def test_mask_result_from_user_masks_validates():
    """User masks are accepted when shapes match, rejected otherwise."""
    good = np.ones((4, 5), dtype=bool)
    result = mask_result_from_user_masks([good], [(4, 5)], MaskParams(True))
    assert result.flat.sum() == 20

    with pytest.raises(ValueError):
        mask_result_from_user_masks([good, good], [(4, 5)], MaskParams(True))

    with pytest.raises(ValueError):
        mask_result_from_user_masks(
            [np.ones((3, 5), dtype=bool)], [(4, 5)], MaskParams(True)
        )
