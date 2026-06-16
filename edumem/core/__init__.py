"""edumem Core - Native SQLite memory implementation"""

from edumem.core.plugins import (
    edumemPlugin,
    PluginManager,
    LoggingPlugin,
    MetricsPlugin,
    FilterPlugin,
    get_manager,
)
from edumem.core.streaming import (
    MemoryStream,
    MemoryEvent,
    EventType,
    DeltaSync,
    SyncCheckpoint,
)
from edumem.core.patterns import (
    MemoryCompressor,
    PatternDetector,
    CompressionStats,
    DetectedPattern,
)

__all__ = [
    # Plugins
    "edumemPlugin",
    "PluginManager",
    "LoggingPlugin",
    "MetricsPlugin",
    "FilterPlugin",
    "get_manager",
    # Streaming
    "MemoryStream",
    "MemoryEvent",
    "EventType",
    "DeltaSync",
    "SyncCheckpoint",
    # Patterns
    "MemoryCompressor",
    "PatternDetector",
    "CompressionStats",
    "DetectedPattern",
]
