"""Exercise packed plane copies, evictions, duplicate IDs and uncached shard misses."""
from array import array
from types import SimpleNamespace
import unittest

from retained_rows import RetainedRows


class NativeReader:
    def __init__(self):
        self.calls = []

    def ple_reader_run(self, handle, ids, weights, scales, count):
        self.calls.append(list(ids))
        for i, row in enumerate(ids):
            valid = 1 <= row < 4
            weights[2*i:2*i+2] = bytes((row, row+10)) if valid else bytes(2)
            if scales is not None:
                scales[i] = 200+row if valid else 0


class CacheTests(unittest.TestCase):
    def test_eviction_preserves_later_hits_and_optional_scales(self):
        for s in (0, 1):
            with self.subTest(scale_bytes=s):
                native = NativeReader()
                reader = SimpleNamespace(weight_row_bytes=2, scale_row_bytes=s, shard_start=1,
                                         shard_end=4, _native=native, _reader=None)
                cache = RetainedRows(reader, 2*(2+s))
                for ids in ([1, 2, 2], [3, 1, 3, 0], [1, 3], []):
                    reader._ids_buffer = memoryview(array("q", ids))
                    reader._weight_buffer = memoryview(bytearray(2*len(ids)))
                    reader._scale_buffer = memoryview(bytearray(s*len(ids))) if s else None
                    cache.gather(reader, len(ids))
                    expected = b"".join(bytes((i, i+10)) if i else bytes(2) for i in ids)
                    self.assertEqual(bytes(reader._weight_buffer), expected)
                    if s:
                        self.assertEqual(bytes(reader._scale_buffer), bytes(200+i if i else 0 for i in ids))
                self.assertEqual(native.calls, [[1, 2], [3, 0]])
                self.assertEqual(len(cache.entries), 2)
                self.assertEqual(cache.hits, 3)


if __name__ == "__main__":
    unittest.main()
