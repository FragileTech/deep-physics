from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_REGISTRY = PROJECT_ROOT / "models"
MODEL_ACTION_REGISTRY = PROJECT_ROOT / "models_action"
METRICS_REGISTRY = PROJECT_ROOT / "mlruns"
