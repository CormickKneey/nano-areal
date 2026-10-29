"""Dependency-free S1 sync regression tests; discover with -p 'test_s1_*.py'."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from nano_areal.engine.buffer import AsyncTrajectoryBuffer


class SyncContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_gap_filters_stale_and_future(self):
        buffer = AsyncTrajectoryBuffer(max_head_offpolicyness=0)
        buffer.update_policy_version(2)
        for version in (1, 3, 2):
            buffer.put_nowait(SimpleNamespace(policy_version=version))
        batch = await buffer.get_batch(1, timeout=0.1)
        self.assertEqual([item.policy_version for item in batch], [2])
        self.assertEqual(buffer.stats.total_discarded, 2)

    async def test_sync_uses_current_version_after_update(self):
        buffer = AsyncTrajectoryBuffer(max_head_offpolicyness=0)
        buffer.put_nowait(SimpleNamespace(policy_version=0))
        buffer.update_policy_version(1)
        buffer.put_nowait(SimpleNamespace(policy_version=1))
        batch = await asyncio.wait_for(buffer.get_batch_sync(1), timeout=0.2)
        self.assertEqual(batch[0].policy_version, 1)
        self.assertEqual(buffer.stats.total_discarded, 1)

    async def test_async_boundary_and_incomplete_batch(self):
        buffer = AsyncTrajectoryBuffer(max_head_offpolicyness=2)
        buffer.update_policy_version(4)
        for version in (1, 2, 4, 5):
            buffer.put_nowait(SimpleNamespace(policy_version=version))
        batch = await buffer.get_batch(3, timeout=0.02)
        self.assertEqual([item.policy_version for item in batch], [2, 4])
        self.assertEqual(buffer.stats.total_discarded, 2)


if __name__ == "__main__":
    unittest.main()
