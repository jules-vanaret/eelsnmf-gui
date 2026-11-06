from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg", force=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
