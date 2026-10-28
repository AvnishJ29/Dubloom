"""Concurrency must overlap requests without changing ordered segment data."""

import threading
import unittest
from backend.jobs.pipeline import run_segment_stage


class ParallelStagesTest(unittest.TestCase):
    def test_requests_overlap_and_keep_segment_order(self):
        barrier = threading.Barrier(3, timeout=5)
        segments = [{'index': i} for i in range(3)]
        completed = []

        def worker(index, item):
            barrier.wait()
            item['result'] = index * 10

        run_segment_stage(segments, worker, 3, completed.append)
        self.assertEqual([item['result'] for item in segments], [0, 10, 20])
        self.assertEqual(completed, [1, 2, 3])

    def test_failure_is_not_hidden(self):
        def fail(index, item):
            raise RuntimeError('provider failed')

        with self.assertRaisesRegex(RuntimeError, 'provider failed'):
            run_segment_stage([1], fail, 1, lambda count: None)
