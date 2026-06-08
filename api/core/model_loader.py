import pickle
import sys
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.train import SoftEnsemble, XGBWithWeight, LGBMWithWeight  # noqa

class _Unpickler(pickle.Unpickler):
    """Redirects __main__.ClassName → model.train.ClassName"""
    def find_class(self, module, name):
        if module == "__main__":
            module = "model.train"
        return super().find_class(module, name)

_model_cache   = None
_poisson_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        from .config import MODEL_PATH
        with open(MODEL_PATH, "rb") as f:
            _model_cache = _Unpickler(f).load()
    return _model_cache

def get_poisson():
    global _poisson_cache
    if _poisson_cache is None:
        from .config import POISSON_PATH
        with open(POISSON_PATH, "rb") as f:
            _poisson_cache = pickle.load(f)
    return _poisson_cache