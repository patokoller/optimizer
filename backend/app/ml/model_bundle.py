"""
Model-bundle persistence: save the fitted scoring models from a run and reload
them later for on-demand single-ticker scoring (no retraining).

Why this exists
---------------
A full run trains three ensembles (~22 min, dominated by entropy) and then throws
them away. On-demand scoring of an arbitrary US-listed ticker cannot afford to
retrain, so we persist the fitted models PLUS each strategy's universe
raw-ensemble vector. The raw vectors are the reference distributions that let a
new ticker's raw prediction be ranked into the same cross-sectional scale the run
used — see `scoring.percentile_into`.

Storage: a single Postgres row per bundle (pickled models in a bytea column).
Postgres is used rather than the filesystem because Railway's FS is ephemeral
across redeploys; a bundle must survive them.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app import models

logger = logging.getLogger(__name__)

# Keep only the most recent N bundles; older ones are pruned on each save.
_MAX_BUNDLES = 5


def _capture_lib_versions() -> dict[str, str]:
    """Record the ML lib versions present at save time, for load-time mismatch
    warnings (a model pickled under one major version may misbehave under
    another). Absent libs are simply skipped."""
    out: dict[str, str] = {}
    for lib in ("sklearn", "xgboost", "lightgbm", "catboost", "numpy"):
        try:
            out[lib] = __import__(lib).__version__
        except Exception:
            pass
    return out


@dataclass
class LoadedBundle:
    """Unpickled bundle ready for on-demand scoring."""
    run_id: Optional[str]
    run_type: str
    rebalance_date: Optional[datetime]
    frequency: str
    universe: list[str]
    strategies: list[str]
    models: dict[str, Any]                       # {strategy: fitted_model}
    raw_vectors: dict[str, dict[str, float]]     # {strategy: {ticker: raw_ensemble}}
    lib_versions: dict[str, str]
    created_at: Optional[datetime]

    def reference_raw(self, strategy: str) -> list[float]:
        """The universe raw-ensemble values for a strategy, as the reference
        distribution to rank a new ticker into via scoring.percentile_into."""
        vec = self.raw_vectors.get(strategy) or {}
        return [float(v) for v in vec.values() if v is not None]


def save_bundle(
    db,
    *,
    run_id: Optional[str],
    run_type: str,
    rebalance_date: Optional[datetime],
    frequency: str,
    universe: list[str],
    models_by_strategy: dict[str, Any],
    score_dicts_by_strategy: dict[str, dict[str, dict]],
) -> Optional[str]:
    """
    Persist the fitted models from a run.

    models_by_strategy:        {strategy: fitted_model_or_None}
    score_dicts_by_strategy:   {strategy: predict_output_dict}  (per-ticker dicts
                               that now carry "raw_ensemble") — used to build the
                               reference distributions.

    Only strategies whose model is present AND trained are stored. Returns the
    new bundle id, or None if nothing was persistable.
    """
    # Keep only trained models.
    present: dict[str, Any] = {}
    for strat, model in models_by_strategy.items():
        if model is None:
            continue
        if not getattr(model, "_trained", False):
            logger.info(f"Bundle: skipping {strat} — model not trained")
            continue
        present[strat] = model

    if not present:
        logger.warning("Bundle: no trained models to persist — skipping save")
        return None

    # Build reference raw-ensemble vectors for the present strategies.
    raw_vectors: dict[str, dict[str, float]] = {}
    for strat in present:
        sd = score_dicts_by_strategy.get(strat) or {}
        vec = {
            t: float(d["raw_ensemble"])
            for t, d in sd.items()
            if isinstance(d, dict) and d.get("raw_ensemble") is not None
        }
        raw_vectors[strat] = vec
        if not vec:
            logger.warning(
                f"Bundle: {strat} has no raw_ensemble values — percentile_into "
                f"will fall back to 0.5 for on-demand scoring of this strategy"
            )

    try:
        blob = pickle.dumps(present, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Bundle: pickling failed ({e}) — skipping save")
        return None

    row = models.ModelBundle(
        run_id=run_id,
        run_type=run_type,
        rebalance_date=rebalance_date,
        frequency=frequency,
        universe=list(universe),
        strategies=sorted(present.keys()),
        raw_vectors=raw_vectors,
        lib_versions=_capture_lib_versions(),
        blob=blob,
        blob_bytes=len(blob),
    )
    db.add(row)
    db.commit()
    logger.info(
        f"Bundle saved {row.id} — strategies={sorted(present.keys())} "
        f"universe={len(universe)} blob={len(blob)/1024:.0f}KB"
    )

    _prune(db)
    return row.id


def load_latest_bundle(
    db, *, require_strategies: Optional[list[str]] = None
) -> Optional[LoadedBundle]:
    """
    Load the most recent bundle (by created_at). If require_strategies is given,
    return the most recent bundle that contains ALL of them.

    Logs a warning when the saved lib versions differ from the current
    environment, since that can silently affect predictions.
    """
    q = db.query(models.ModelBundle).order_by(models.ModelBundle.created_at.desc())
    row = None
    for candidate in q.limit(_MAX_BUNDLES + 5).all():
        if require_strategies and not set(require_strategies).issubset(
            set(candidate.strategies or [])
        ):
            continue
        row = candidate
        break

    if row is None:
        logger.info("Bundle: no persisted bundle found"
                    + (f" with strategies {require_strategies}" if require_strategies else ""))
        return None

    try:
        loaded_models = pickle.loads(row.blob)
    except Exception as e:
        logger.error(f"Bundle: unpickling {row.id} failed ({e})")
        return None

    current = _capture_lib_versions()
    saved = row.lib_versions or {}
    drift = {k: (saved.get(k), current.get(k)) for k in saved
             if k in current and saved.get(k) != current.get(k)}
    if drift:
        logger.warning(f"Bundle {row.id}: lib version drift since save — {drift}")

    return LoadedBundle(
        run_id=row.run_id,
        run_type=row.run_type,
        rebalance_date=row.rebalance_date,
        frequency=row.frequency or "monthly",
        universe=list(row.universe or []),
        strategies=list(row.strategies or []),
        models=loaded_models,
        raw_vectors=row.raw_vectors or {},
        lib_versions=saved,
        created_at=row.created_at,
    )


def _prune(db) -> None:
    """Delete all but the most recent _MAX_BUNDLES bundles."""
    try:
        stale = (
            db.query(models.ModelBundle)
            .order_by(models.ModelBundle.created_at.desc())
            .offset(_MAX_BUNDLES)
            .all()
        )
        if stale:
            for s in stale:
                db.delete(s)
            db.commit()
            logger.info(f"Bundle: pruned {len(stale)} old bundle(s)")
    except Exception as e:
        logger.warning(f"Bundle: prune failed ({e})")
        db.rollback()
