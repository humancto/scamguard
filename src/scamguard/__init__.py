"""ScamGuard public SDK."""

from .scanner import Scanner, scan
from .schema import ScanResult

__all__ = ["ScanResult", "Scanner", "scan"]
__version__ = "0.1.0"
