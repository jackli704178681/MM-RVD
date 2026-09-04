from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CEBRAFitResult:
    fit_succeeded: bool
    fit_method: str
    error: str = ""


def filter_supported_cebra_kwargs(
    requested_kwargs: dict[str, Any],
    *,
    supported_parameter_names: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter CEBRA constructor kwargs while preserving an audit trail."""
    if supported_parameter_names is None:
        try:
            import cebra  # type: ignore

            signature = inspect.signature(cebra.CEBRA)
            supported_parameter_names = set(signature.parameters)
        except Exception:
            supported_parameter_names = set()

    accepts_any_kwargs = "kwargs" in supported_parameter_names or any(
        name.startswith("**") for name in supported_parameter_names
    )
    if accepts_any_kwargs:
        supported = dict(requested_kwargs)
    else:
        supported = {k: v for k, v in requested_kwargs.items() if k in supported_parameter_names}

    record = {
        "requested": sorted(requested_kwargs),
        "supported_by_api": sorted(supported),
        "not_supported_by_api": sorted(k for k in requested_kwargs if k not in supported),
    }
    return supported, record


def requested_cebra_kwargs(cfg: Any) -> dict[str, Any]:
    device = getattr(cfg, "device", "cuda_if_available")
    requested_device = "cuda" if device == "cuda_if_available" else device
    return {
        "model_architecture": "offset10-model",
        "batch_size": int(getattr(cfg, "batch_size", 512)),
        "learning_rate": float(getattr(cfg, "learning_rate", 3e-4)),
        "output_dimension": int(getattr(cfg, "output_dim", 16)),
        "max_iterations": int(getattr(cfg, "max_iterations", 2000)),
        "distance": getattr(cfg, "distance", "cosine"),
        "conditional": getattr(cfg, "conditional", "time_delta"),
        "temperature": float(getattr(cfg, "temperature", 1.0)),
        "device": requested_device,
        "verbose": False,
    }


def create_cebra_model(cfg: Any) -> tuple[Any, dict[str, Any]]:
    import cebra  # type: ignore

    requested = requested_cebra_kwargs(cfg)
    supported, record = filter_supported_cebra_kwargs(requested)
    try:
        model = cebra.CEBRA(**supported)
        record["constructor_status"] = "created"
        record["constructor_error"] = ""
        record["actual_kwargs"] = supported
        return model, record
    except TypeError as exc:
        fallback_keys = ["batch_size", "learning_rate", "output_dimension", "max_iterations"]
        fallback = {k: supported[k] for k in fallback_keys if k in supported}
        model = cebra.CEBRA(**fallback)
        record["constructor_status"] = "created_with_minimal_fallback"
        record["constructor_error"] = str(exc)
        record["actual_kwargs"] = fallback
        record["not_supported_by_api"] = sorted(set(record["not_supported_by_api"]) | set(supported) - set(fallback))
        return model, record


def try_fit_supervised_cebra(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> CEBRAFitResult:
    attempts = [
        ("fit_X_y", lambda: model.fit(X_train, y_train)),
        ("fit_discrete_label", lambda: model.fit(X_train, discrete_label=y_train)),
        ("fit_y_keyword", lambda: model.fit(X_train, y=y_train)),
        ("fit_continuous_label", lambda: model.fit(X_train, continuous_label=y_train)),
    ]
    errors: list[str] = []
    for name, call in attempts:
        try:
            call()
            return CEBRAFitResult(True, name)
        except TypeError as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return CEBRAFitResult(False, "failed_api_incompatible", " | ".join(errors))


def transform_cebra(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "transform"):
        return np.asarray(model.transform(X))
    raise AttributeError("CEBRA model has no transform method")
