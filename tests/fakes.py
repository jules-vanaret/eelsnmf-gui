"""Shared synthetic data and test doubles for the eelsnmf-gui test suite."""

from __future__ import annotations

import os
from collections.abc import Sequence
from types import SimpleNamespace
from typing import List

import numpy as np

from eelsnmf_gui._analysis import find_index
from eelsnmf_gui._results import (
    BeamParams,
    DecoBundle,
    PreprocessedSI,
)

ENERGY_OFFSET = 500.0
ENERGY_SCALE = 2.0
N_ENERGY = 32
NY = 4
NX = 5


def synthetic_data(ny=NY, nx=NX, n_energy=N_ENERGY, seed=0):
    """A cube with two Gaussian peaks at fixed energies plus small noise."""
    rng = np.random.default_rng(seed)
    axis = ENERGY_OFFSET + ENERGY_SCALE * np.arange(n_energy)
    yy, xx = np.mgrid[0:ny, 0:nx]
    weight_a = (xx + 1) / (nx + 1)
    weight_b = 1 - weight_a
    spectrum_a = np.exp(-0.5 * ((axis - 530.0) / 6.0) ** 2)
    spectrum_b = np.exp(-0.5 * ((axis - 555.0) / 6.0) ** 2)
    data = weight_a[..., None] * spectrum_a + weight_b[..., None] * spectrum_b
    data = data + 0.01 * rng.normal(size=data.shape)
    return data.astype(np.float32)


def make_si(label, key=None, ny=NY, nx=NX, seed=0):
    """A PreprocessedSI with synthetic data (raw is None for stub engines)."""
    if key is None:
        key = os.path.join("/fake", label)
    data = synthetic_data(ny=ny, nx=nx, seed=seed)
    return PreprocessedSI(
        key=key,
        label=label,
        data=data,
        raw=None,
        original_image=data.mean(-1),
        pixel_scale=0.2,
        energy_offset=ENERGY_OFFSET,
        energy_scale=ENERGY_SCALE,
    )


class FakeLoader:
    """Stands in for the EML loader; returns synthetic SIs and counts calls."""

    def __init__(self, n_sis_per_folder=1, ny=NY, nx=NX, reverse=False):
        self.n_sis_per_folder = n_sis_per_folder
        self.ny = ny
        self.nx = nx
        self.reverse = reverse
        self.calls = 0
        self.last_folders = None
        self.last_labels = None
        self.last_median_filter_size = None

    def __call__(self, folders, labels, median_filter_size):
        self.calls += 1
        self.last_folders = list(folders)
        self.last_labels = list(labels)
        self.last_median_filter_size = median_filter_size

        pairs = list(zip(folders, labels))
        if self.reverse:
            pairs = pairs[::-1]

        sis: List[PreprocessedSI] = []
        for folder, label in pairs:
            for j in range(self.n_sis_per_folder):
                suffix = f"_{j}" if self.n_sis_per_folder > 1 else ""
                seed = sum(ord(c) for c in label) + j
                sis.append(
                    make_si(
                        label + suffix,
                        key=folder,
                        ny=self.ny,
                        nx=self.nx,
                        seed=seed,
                    )
                )
        return sis


class StubSl:
    """Stand-in for an EELSNMF ListOfSI wrapper."""

    def __init__(self, shapes: Sequence):
        self.shapes = [tuple(shape) for shape in shapes]
        self.spatial_dim_list = [h * w for h, w in self.shapes]

    def fold_array(self, vector):
        vector = np.asarray(vector)
        images = []
        offset = 0
        for height, width in self.shapes:
            count = height * width
            images.append(vector[offset : offset + count].reshape(height, width))
            offset += count
        return images


class StubDeco:
    """Stand-in for a fitted EELSNMF decomposition object."""

    def __init__(self, shapes: Sequence, energy_axis: np.ndarray, mask_flat=None):
        self.shapes = [tuple(shape) for shape in shapes]
        self.n_pix = sum(h * w for h, w in self.shapes)
        self.mask_flat = mask_flat
        # A masked decomposition works on masked pixels only.
        self.n_data = (
            self.n_pix if mask_flat is None else int(np.sum(mask_flat))
        )
        self.energy_axis = np.asarray(energy_axis, dtype=float)
        self.n_energy = len(self.energy_axis)
        self.fine_structure_ranges = {}
        self.model = SimpleNamespace(_edge_slices={}, backgrounds=None)
        self.X = np.ones((self.n_energy, self.n_data))
        self.G = np.ones((self.n_energy, 4))
        self.W = np.ones((4, 2))
        self.H = np.ones((2, self.n_data))
        self.n_components = 2
        self._edges = {}
        self.plot_calls = []
        # Attributes recoverable by the GUI when loading a deco.
        self.spatial_shape = (
            self.shapes[0] if len(self.shapes) == 1 else (self.n_pix,)
        )
        self.E0 = 300_000.0
        self.alpha = 0.0154
        self.beta = 0.01846
        self.max_iters = 100
        self.tol = 1e-8
        self.error_skip_step = 100
        self.analysis_description = {"decomposition": {}}

    def get_model(self):
        return np.full((self.n_energy, self.n_data), 0.1)

    def get_edge_from_component(self, comp_idx, edge_label):
        return SimpleNamespace(data=self._edges[edge_label][comp_idx])

    def save(self, fname, path=None, save_hspy_objects=False, overwrite=False):
        """Mimic EELSNMF's save: pickle the deco into path/fname.pkl."""
        import os
        import pickle as pkl

        if path is None:
            path = os.getcwd()
        target = os.path.join(path, fname + ".pkl")
        if os.path.exists(target) and not overwrite:
            print("File already exists")
            return
        with open(target, "wb") as fh:
            pkl.dump(self, fh)

    def plot_factors(self):
        self.plot_calls.append("plot_factors")


class StubEngine:
    """Deterministic engine double with two components of fixed power."""

    def __init__(self, powers=(1.0, 0.25)):
        self.powers = list(powers)
        self.build_deco_calls = []
        self.build_g_calls = []
        self.decompose_calls = []
        self.list_of_si_calls = 0

    def list_of_si(self, sis, energy_axis):
        self.list_of_si_calls += 1
        return StubSl([si.spatial_shape for si in sis])

    def build_deco(
        self, sis, energy_axis, *, E0_V, alpha_rad, beta_rad, mask_flat=None
    ):
        self.build_deco_calls.append(
            {
                "keys": [si.key for si in sis],
                "n_sis": len(sis),
                "energy_axis": np.asarray(energy_axis, dtype=float),
                "E0_V": E0_V,
                "mask_pixels": (
                    None if mask_flat is None else int(np.sum(mask_flat))
                ),
            }
        )
        shapes = [si.spatial_shape for si in sis]
        deco = StubDeco(shapes, energy_axis, mask_flat=mask_flat)
        return DecoBundle(deco=deco, list_of_si=StubSl(shapes))

    def build_g(self, bundle, fine_structure_ranges, *, n_power_laws=50):
        deco = bundle.deco
        self.build_g_calls.append(
            {
                "fine_structure_ranges": dict(fine_structure_ranges),
                "n_power_laws": n_power_laws,
            }
        )
        deco.fine_structure_ranges = dict(fine_structure_ranges)

        # One basis column per edge channel (delta spectrum placed at the
        # channel energy), plus a few smooth background columns. Edge slices
        # address the *basis* axis of G/W, as in EELSNMF.
        slices = {}
        spans = {}
        for label, (lo, hi) in fine_structure_ranges.items():
            ii, ff = find_index(deco.energy_axis, (lo, hi))
            spans[label] = (int(ii), max(int(ff), int(ii) + 1))
        edge_cols = sum(hi - ii for ii, hi in spans.values())
        n_bg = 3
        G = np.zeros((deco.n_energy, edge_cols + n_bg))
        col = 0
        for label, (ii, ff) in spans.items():
            slices[label] = slice(col, col + (ff - ii))
            for offset in range(ff - ii):
                G[min(ii + offset, deco.n_energy - 1), col] = 1.0
                col += 1
        G[:, col:] = 0.1
        deco.G = G
        deco.model._edge_slices = slices

    def decompose(
        self,
        bundle,
        *,
        n_components,
        max_iters,
        w_init=None,
        use_cupy=None,
        tol=1e-8,
        error_skip_step=100,
        decomposition_method="default_decomposition",
        regularization=None,
    ):
        deco = bundle.deco
        self.decompose_calls.append(
            {
                "n_components": n_components,
                "max_iters": max_iters,
                "w_init_shape": (
                    None if w_init is None else tuple(w_init.shape)
                ),
                "G_shape": tuple(deco.G.shape),
                "use_cupy": use_cupy,
                "tol": tol,
                "error_skip_step": error_skip_step,
                "decomposition_method": decomposition_method,
                "regularization": regularization,
                "mask_pixels": (
                    None
                    if deco.mask_flat is None
                    else int(np.sum(deco.mask_flat))
                ),
            }
        )
        deco.n_components = n_components
        k = n_components
        powers = self.powers[:k]

        nG = deco.G.shape[1]
        W = np.zeros((nG, k))
        col = 0
        for slc in deco.model._edge_slices.values():
            width = slc.stop - slc.start
            for j, power in enumerate(powers):
                W[col : col + width, j] = power / width
            col += width
        W[col:, :] = 0.01
        deco.W = W

        # A masked decomposition produces H over masked pixels only.
        n_pix = deco.n_data
        deco.H = np.array(
            [
                1.0 + 0.5 * (np.arange(n_pix) % max(k, 1) == j)
                for j in range(k)
            ]
        )
        for label in deco.model._edge_slices:
            base = np.linspace(0.1, 1.0, deco.n_energy)
            deco._edges[label] = [base * (j + 1) for j in range(4)]


RUN_KWARGS = dict(
    beam=BeamParams(E0_V=300_000.0, alpha_rad=0.0154, beta_rad=0.01846),
    fine_structure_ranges={"O_K": (530.0, 550.0), "Fe_L": (554.0, 562.0)},
    n_components=2,
    max_iters=100,
    colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
)
