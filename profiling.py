"""
Profiling module for tracking timing statistics of tools and LLM API calls.

This module provides a centralized way to track timing information for various
operations in the agent system, including:
- Individual tool executions
- OpenAI API calls
- Aggregate statistics (min, max, median, mean, total)
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class ProfilingStats:
    """Statistics for a particular operation type."""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    def add_timing(self, duration: float):
        """Add a timing measurement."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)

    @property
    def mean_time(self) -> float:
        """Calculate mean time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """Calculate median time."""
        return statistics.median(self.times) if self.times else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "call_count": self.call_count,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "mean_time": self.mean_time,
            "median_time": self.median_time
        }


class Profiler:
    """
    Global profiler for tracking timing statistics across tools and API calls.

    Usage:
        profiler = Profiler()

        # Time a tool execution
        with profiler.time_tool("web_search"):
            # ... tool execution code ...
            pass

        # Time an API call
        with profiler.time_api_call():
            # ... API call code ...
            pass

        # Get statistics
        stats = profiler.get_statistics()
    """

    def __init__(self):
        """Initialize the profiler."""
        self.tool_stats: Dict[str, ProfilingStats] = defaultdict(ProfilingStats)
        self.api_stats: ProfilingStats = ProfilingStats()
        self._enabled = True

    def enable(self):
        """Enable profiling."""
        self._enabled = True

    def disable(self):
        """Disable profiling."""
        self._enabled = False

    def reset(self):
        """Reset all profiling data."""
        self.tool_stats.clear()
        self.api_stats = ProfilingStats()

    def record_tool_timing(self, tool_name: str, duration: float):
        """Record timing for a tool execution."""
        if self._enabled:
            self.tool_stats[tool_name].add_timing(duration)

    def record_api_timing(self, duration: float):
        """Record timing for an API call."""
        if self._enabled:
            self.api_stats.add_timing(duration)

    def get_statistics(self) -> Dict:
        """
        Get all profiling statistics.

        Returns:
            Dictionary containing tool and API statistics
        """
        return {
            "tools": {
                tool_name: stats.to_dict()
                for tool_name, stats in sorted(self.tool_stats.items())
            },
            "api_calls": self.api_stats.to_dict()
        }

    def print_statistics(self, detailed: bool = True):
        """
        Print profiling statistics in a readable format.

        Args:
            detailed: If True, show per-tool breakdown. If False, show summary only.
        """
        print("\n" + "="*80)
        print("📊 PROFILING STATISTICS")
        print("="*80)

        # API Call Statistics
        print("\n🔷 OpenAI API Calls:")
        if self.api_stats.call_count > 0:
            api_dict = self.api_stats.to_dict()
            print(f"  Total Calls:  {api_dict['call_count']}")
            print(f"  Total Time:   {api_dict['total_time']:.2f}s")
            print(f"  Min Time:     {api_dict['min_time']:.2f}s")
            print(f"  Max Time:     {api_dict['max_time']:.2f}s")
            print(f"  Mean Time:    {api_dict['mean_time']:.2f}s")
            print(f"  Median Time:  {api_dict['median_time']:.2f}s")
        else:
            print("  No API calls recorded")

        # Tool Statistics
        print("\n🔧 Tool Executions:")
        if self.tool_stats:
            if detailed:
                for tool_name in sorted(self.tool_stats.keys()):
                    stats_dict = self.tool_stats[tool_name].to_dict()
                    print(f"\n  📌 {tool_name}:")
                    print(f"     Total Calls:  {stats_dict['call_count']}")
                    print(f"     Total Time:   {stats_dict['total_time']:.2f}s")
                    print(f"     Min Time:     {stats_dict['min_time']:.2f}s")
                    print(f"     Max Time:     {stats_dict['max_time']:.2f}s")
                    print(f"     Mean Time:    {stats_dict['mean_time']:.2f}s")
                    print(f"     Median Time:  {stats_dict['median_time']:.2f}s")

            # Summary
            total_tool_calls = sum(s.call_count for s in self.tool_stats.values())
            total_tool_time = sum(s.total_time for s in self.tool_stats.values())
            print(f"\n  📊 Summary:")
            print(f"     Total Tool Calls:  {total_tool_calls}")
            print(f"     Total Tool Time:   {total_tool_time:.2f}s")
            print(f"     Unique Tools Used: {len(self.tool_stats)}")
        else:
            print("  No tool executions recorded")

        # Overall Summary
        total_api_time = self.api_stats.total_time
        total_tool_time = sum(s.total_time for s in self.tool_stats.values())
        print(f"\n📈 Overall Summary:")
        print(f"  Total API Time:   {total_api_time:.2f}s")
        print(f"  Total Tool Time:  {total_tool_time:.2f}s")
        print(f"  Total Time:       {total_api_time + total_tool_time:.2f}s")
        print("="*80 + "\n")

    def export_to_json(self) -> str:
        """Export statistics as JSON string."""
        import json
        return json.dumps(self.get_statistics(), indent=2)

    def export_to_file(self, filepath: str):
        """
        Export statistics to a JSON file.

        Args:
            filepath: Path to output file
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        print(f"📁 Profiling statistics exported to: {filepath}")


# Global profiler instance
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def reset_profiler():
    """Reset the global profiler."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, profiler: Profiler, operation_type: str, operation_name: Optional[str] = None):
        """
        Initialize timing context.

        Args:
            profiler: Profiler instance to record timing
            operation_type: 'tool' or 'api'
            operation_name: Name of the operation (required for tools)
        """
        self.profiler = profiler
        self.operation_type = operation_type
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        duration = time.time() - self.start_time

        if self.operation_type == 'tool':
            self.profiler.record_tool_timing(self.operation_name, duration)
        elif self.operation_type == 'api':
            self.profiler.record_api_timing(duration)

        return False  # Don't suppress exceptions


def aggregate_profiling_stats(stats_list: List[Dict]) -> Dict:
    """
    Aggregate multiple profiling statistics dictionaries into one.

    This is useful for batch processing where each worker process has its own
    profiler instance that needs to be combined.

    Args:
        stats_list: List of statistics dictionaries from get_statistics()

    Returns:
        Dict: Aggregated statistics with combined tool and API call data
    """
    # Note: per-call timings are not preserved across worker boundaries, so we
    # combine the per-conversation summary stats directly. This gives correct
    # call_count/total_time/min/max/mean. ``median`` cannot be reconstructed
    # exactly from summaries; we surface mean as a best-effort approximation
    # and flag it via the ``median_time_approximate`` field.

    def _empty():
        return {
            "call_count": 0,
            "total_time": 0.0,
            "min_time": float("inf"),
            "max_time": 0.0,
        }

    tool_acc: Dict[str, Dict] = defaultdict(_empty)
    api_acc = _empty()

    def _merge(acc: Dict, summary: Dict) -> None:
        count = summary.get("call_count", 0)
        if count <= 0:
            return
        acc["call_count"] += count
        acc["total_time"] += summary.get("total_time", 0.0)
        # Only consider min/max if the source actually had calls; otherwise
        # its min_time will be the sentinel 0.0 from to_dict().
        acc["min_time"] = min(acc["min_time"], summary.get("min_time", float("inf")))
        acc["max_time"] = max(acc["max_time"], summary.get("max_time", 0.0))

    for stats in stats_list:
        for tool_name, tool_stats in stats.get("tools", {}).items():
            _merge(tool_acc[tool_name], tool_stats)
        _merge(api_acc, stats.get("api_calls", {}))

    def _finalize(acc: Dict) -> Dict:
        count = acc["call_count"]
        if count == 0:
            return {
                "call_count": 0,
                "total_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "mean_time": 0.0,
                "median_time": 0.0,
                "median_time_approximate": True,
            }
        mean_time = acc["total_time"] / count
        return {
            "call_count": count,
            "total_time": acc["total_time"],
            "min_time": acc["min_time"] if acc["min_time"] != float("inf") else 0.0,
            "max_time": acc["max_time"],
            "mean_time": mean_time,
            # Real median requires per-call data we don't carry across workers.
            "median_time": mean_time,
            "median_time_approximate": True,
        }

    final_stats = {
        "tools": {name: _finalize(acc) for name, acc in tool_acc.items()},
        "api_calls": _finalize(api_acc),
    }

    return final_stats


def print_aggregated_statistics(stats: Dict, detailed: bool = True):
    """
    Print aggregated profiling statistics in a readable format.

    Args:
        stats: Aggregated statistics dictionary from aggregate_profiling_stats()
        detailed: If True, show per-tool breakdown. If False, show summary only.
    """
    print("\n" + "="*80)
    print("📊 AGGREGATED PROFILING STATISTICS")
    print("="*80)

    # API Call Statistics
    print("\n🔷 OpenAI API Calls:")
    api_stats = stats.get("api_calls", {})
    if api_stats.get("call_count", 0) > 0:
        print(f"  Total Calls:  {api_stats['call_count']}")
        print(f"  Total Time:   {api_stats['total_time']:.2f}s")
        print(f"  Min Time:     {api_stats['min_time']:.2f}s")
        print(f"  Max Time:     {api_stats['max_time']:.2f}s")
        print(f"  Mean Time:    {api_stats['mean_time']:.2f}s")
        print(f"  Median Time:  {api_stats['median_time']:.2f}s")
    else:
        print("  No API calls recorded")

    # Tool Statistics
    print("\n🔧 Tool Executions:")
    tool_stats = stats.get("tools", {})
    if tool_stats:
        if detailed:
            for tool_name in sorted(tool_stats.keys()):
                stats_dict = tool_stats[tool_name]
                print(f"\n  📌 {tool_name}:")
                print(f"     Total Calls:  {stats_dict['call_count']}")
                print(f"     Total Time:   {stats_dict['total_time']:.2f}s")
                print(f"     Min Time:     {stats_dict['min_time']:.2f}s")
                print(f"     Max Time:     {stats_dict['max_time']:.2f}s")
                print(f"     Mean Time:    {stats_dict['mean_time']:.2f}s")
                print(f"     Median Time:  {stats_dict['median_time']:.2f}s")

        # Summary
        total_tool_calls = sum(s["call_count"] for s in tool_stats.values())
        total_tool_time = sum(s["total_time"] for s in tool_stats.values())
        print(f"\n  📊 Summary:")
        print(f"     Total Tool Calls:  {total_tool_calls}")
        print(f"     Total Tool Time:   {total_tool_time:.2f}s")
        print(f"     Unique Tools Used: {len(tool_stats)}")
    else:
        print("  No tool executions recorded")

    # Overall Summary
    total_api_time = api_stats.get("total_time", 0.0)
    total_tool_time = sum(s["total_time"] for s in tool_stats.values())
    print(f"\n📈 Overall Summary:")
    print(f"  Total API Time:   {total_api_time:.2f}s")
    print(f"  Total Tool Time:  {total_tool_time:.2f}s")
    print(f"  Total Time:       {total_api_time + total_tool_time:.2f}s")
    print("="*80 + "\n")
