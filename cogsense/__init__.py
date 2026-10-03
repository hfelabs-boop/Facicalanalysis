"""CogSense HFE Module — Facial Action Coding & Micro-Expression Cognitive Engine.

Extracts upper-face Action Units (AU1, AU2, AU4, AU7, AU14, AU43/45) from a
478-point face mesh, derives a Mental Effort Score (MES) and Cognitive
Friction Index (CFI), and fuses them with gaze AOIs and RULA posture scores.
"""

from cogsense.config import CogSenseConfig
from cogsense.engine import CogSenseEngine, FaceObservation

__version__ = "1.0.0"

__all__ = ["CogSenseConfig", "CogSenseEngine", "FaceObservation", "__version__"]
