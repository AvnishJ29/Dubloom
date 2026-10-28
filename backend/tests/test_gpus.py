"""Two render slots on two GPUs: each holder's workers see their own card; one GPU is shared when only one exists."""

import threading
import time
import unittest

from backend.jobs.gpus import GpuPool


class GpuPoolTest(unittest.TestCase):
    def test_two_holders_get_two_cards(self):
        pool = GpuPool(["0", "1"])
        seen, ready, go = (
            {},
            threading.Barrier(3),
            threading.Event(),
        )  # two holders and the test thread

        def hold(name):
            pool.acquire()
            seen[name] = pool.env({})["CUDA_VISIBLE_DEVICES"]
            ready.wait()
            go.wait()
            pool.release()

        threads = [threading.Thread(target=hold, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        ready.wait()
        self.assertEqual(set(seen.values()), {"0", "1"})
        self.assertFalse(pool.acquire(blocking=False))  # both slots taken
        go.set()
        for t in threads:
            t.join()
        self.assertTrue(pool.acquire(blocking=False))
        pool.release()

    def test_without_a_slot_the_environment_is_untouched(self):
        pool = GpuPool(["0"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", pool.env({"PATH": "x"}))
        self.assertEqual(pool.env({"PATH": "x"})["PATH"], "x")


class BlockingAcquireTest(unittest.TestCase):
    def test_a_blocking_acquire_waits_for_the_slot(self):
        pool = GpuPool(["0"])
        held = threading.Event()

        def hold():  # a render that finishes shortly
            pool.acquire()
            held.set()
            time.sleep(0.2)
            pool.release()

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait()
        self.assertTrue(pool.acquire())  # waits for the release instead of failing at once
        pool.release()
        holder.join()

    def test_a_timeout_gives_up_while_the_slot_is_held(self):
        pool = GpuPool(["0"])
        self.assertTrue(pool.acquire(blocking=False))
        self.assertFalse(pool.acquire(timeout=0.1))
        pool.release()


if __name__ == "__main__":
    unittest.main()
