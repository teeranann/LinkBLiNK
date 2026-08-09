"""LinkBLiNK — SMLM particle detection, tracking, and re-identification.

Typical use::

    from linkblink import Config, Pipeline

    config = Config.load("configs/default.yaml")
    Pipeline(config).run_batch()

Individual stages are importable on their own, so a downstream project can
reuse (say) the detector or the judge without running the whole pipeline::

    from linkblink.detection import UNetDetector
    from linkblink.linking import apply_judge
"""

from .config import Config
from .pipeline import Pipeline

__version__ = "2.0.0"

__all__ = ["Config", "Pipeline", "__version__"]
