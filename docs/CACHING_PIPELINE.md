# Caching Pipeline Wiring

## Overview
A single dict cache is passed through the pipeline and is filled incrementally by any UI action. Cache entries are keyed by a prefix plus the minimal parameter set required for that stage.

## Cache Layers and Keys
- preprocess
  - Key: ("preprocess", scope, folders, median_filter_size)
  - Value: PreprocessResult
  - Example: ("preprocess", "individual", "/data/runA", 11)
- mask
  - Key: ("mask", preprocess_key, mask_dilation_iters, mask_mode)
  - Value: mask_flat (np.ndarray)
  - Example: ("mask", ("preprocess", "individual", "/data/runA", 11), 2, "mean")
- run
  - Key: ("run", scope, folders, E0_V, alpha_rad, beta_rad, max_iters, median_filter_size, ranges_key, n_components, mask_enabled, mask_dilation_iters, mask_mode, colors_key, W_init_signature)
  - Value: ProcessingResult
  - Example: ("run", "individual", "/data/runA", 300000.0, 0.0154, 0.01846, 2000, 11, (("Fe_L", 700.0, 740.0),), 2, True, 2, "mean", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), None)

## Adding a New Processing Function
1. Decide the highest stage you need (preprocess, mask, decomposition, full run).
2. Always call _get_preprocess_result(...) to obtain processed_cl and energy_axis.
3. If masking is needed, call _get_mask_flat_from_cache(...) with the preprocess_key and mask params.
4. Build downstream results (deco, plots, exports) using the cached results.
5. If the function produces a reusable final result, store it under a new cache key prefix or reuse the run key pattern.

Example
```python
def build_quick_stat(
  folder: str,
  *,
  median_filter_size: int,
  mask_enabled: bool,
  mask_dilation_iters: int,
  mask_mode: str,
  preprocess_cache: dict,
) -> float:
  preprocess_key = ("preprocess", "individual", folder, median_filter_size)
  preprocess = _get_preprocess_result(
    cache_key=preprocess_key,
    source_folders=[folder],
    labels=[os.path.basename(folder)],
    median_filter_size=median_filter_size,
    preprocess_cache=preprocess_cache,
  )
  mask_flat = _get_mask_flat_from_cache(
    preprocess_key=preprocess_key,
    preprocess_result=preprocess,
    mask_enabled=mask_enabled,
    mask_dilation_iters=mask_dilation_iters,
    mask_mode=mask_mode,
    preprocess_cache=preprocess_cache,
  )
  if mask_flat is None:
    return float(np.mean(preprocess.processed_cl[0].data))
  return float(np.mean(preprocess.processed_cl[0].data[mask_flat]))
```

## Wiring a New GUI Widget
1. Add preprocess_cache to the function call that the widget triggers.
2. Use cache keys that include only the parameters that affect that stage.
3. Do not duplicate preprocessing or mask computation logic in the widget.
4. Invalidate the cache only when parameters that affect the cached stage change.

Example (widget wiring)
```python
def _plot_quick_stat(self):
  value = build_quick_stat(
    self.source_folders[0],
    median_filter_size=self.median_filter_size,
    mask_enabled=self.mask_enabled,
    mask_dilation_iters=self.mask_dilation_iters,
    mask_mode="mean",
    preprocess_cache=self._preprocess_cache,
  )
  print(f"Quick stat: {value:.4f}")
```

## Parameter Invalidation Rules (update when adding new params)
- Changing median_filter_size or folders invalidates preprocess, mask, and run caches.
- Changing mask params invalidates mask and run caches; preprocess remains valid.
- Changing decomposition params (E0, alpha, beta, max_iters, n_components, fine_structure_ranges, W_init) invalidates run caches; preprocess and mask remain valid.
- Changing colors invalidates run caches only.
