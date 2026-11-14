"""
tests/test_watchdog.py — Unit Tests for Resource Watchdog

Tests:
- Process sampling across tracked PIDs.
- Peak aggregate RSS calculation.
- Threshold breach detection and callback triggering.
"""

import os
import time
import unittest

from src.dask_matrix.watchdog import (
    ResourceWatchdog,
    ProcessMemorySample,
)


class TestWatchdog(unittest.TestCase):

    def test_watchdog_sampling(self):
        current_pid = os.getpid()
        watchdog = ResourceWatchdog(
            pids_to_track=[current_pid],
            rss_ceiling_mib=10000.0,
            host_floor_mib=10.0,
            sample_interval_s=0.05,
        )

        with watchdog:
            time.sleep(0.15)

        self.assertGreaterEqual(len(watchdog.samples), 2)
        self.assertGreater(watchdog.peak_process_sum_mib, 0.0)
        self.assertFalse(watchdog.breach_detected)

    def test_watchdog_threshold_breach(self):
        current_pid = os.getpid()
        breach_called = []

        def breach_callback(reason: str, sample: ProcessMemorySample):
            breach_called.append(reason)

        # Set ceiling to 0.1 MiB so current process immediately triggers breach
        watchdog = ResourceWatchdog(
            pids_to_track=[current_pid],
            rss_ceiling_mib=0.1,
            host_floor_mib=10.0,
            sample_interval_s=0.05,
            on_breach_callback=breach_callback,
        )

        with watchdog:
            time.sleep(0.15)

        self.assertTrue(watchdog.breach_detected)
        self.assertIn("Aggregate RSS ceiling breached", watchdog.breach_reason)
        self.assertEqual(len(breach_called), 1)


if __name__ == "__main__":
    unittest.main()
