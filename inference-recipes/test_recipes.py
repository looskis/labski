"""Offline regression checks for rank construction and streaming measurement."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench
import spark


class RecipeTests(unittest.TestCase):
    def test_all_ranks_and_pins(self):
        for name, path in spark.RECIPES.items():
            r = json.loads(path.read_text())
            self.assertRegex(spark.IMAGES[r["image"]], r"@sha256:[0-9a-f]{64}$")
            for ckpt in spark.checkpoints(r):
                self.assertRegex(ckpt["revision"], r"^[0-9a-f]{40}$")
            for n in r["nodes"]:
                for rank in range(n):
                    with self.subTest(recipe=name, nodes=n, rank=rank):
                        args = SimpleNamespace(recipe=name, nodes=n, rank=rank, port=8000,
                                               master_port=29501, head="192.0.2.1" if n > 1 else None,
                                               node_ip=f"192.0.2.{rank + 1}" if n > 1 else None,
                                               interface="cx0" if n > 1 else None, hcas="rdma0,rdma1" if n > 1 else None,
                                               transport=None)
                        cmd = spark.serve_command(args, r, Path("/tmp/test cache"))
                        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], str(n))
                        self.assertEqual("--headless" in cmd, rank > 0 and r.get("engine") != "sglang")
                        self.assertEqual("--host" in cmd, rank == 0 or r.get("engine") == "sglang")
                        self.assertEqual("--device" in cmd, n > 1)
                        if "draft" in r:
                            cfg = json.loads(cmd[cmd.index("--speculative-config") + 1])
                            self.assertTrue(cfg["model"].endswith(r["draft"]["revision"]))

    def test_matched_precision_controls(self):
        bf = json.loads(spark.RECIPES["qwen38-27b-bf16"].read_text())
        fp4 = json.loads(spark.RECIPES["qwen38-27b-nvfp4"].read_text())
        for field in ("image", "context", "sequences", "memory_fraction", "options"):
            self.assertEqual(bf[field], fp4[field])

    def test_routed_tcp_rank(self):
        name = "qwen38-27b-nvfp4"
        recipe = json.loads(spark.RECIPES[name].read_text())
        args = SimpleNamespace(recipe=name, nodes=4, rank=0, port=8000, master_port=29501,
                               head="192.0.2.1", node_ip="192.0.2.1", interface="fabric0",
                               transport="tcp", hcas=None)
        cmd = spark.serve_command(args, recipe, Path("/tmp/cache"))
        self.assertIn("NCCL_NET=Socket", cmd)
        self.assertIn("NCCL_SOCKET_IFNAME==fabric0", cmd)
        self.assertNotIn("/dev/infiniband", cmd)
        args.hcas = "rdma0"
        with self.assertRaises(ValueError):
            spark.serve_command(args, recipe, Path("/tmp/cache"))


class StreamingTests(unittest.TestCase):
    def response(self, done=True, usage=True):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning": "thinking"}}]},
            {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]},
        ]
        if usage:
            events.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        wire = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        if done:
            wire += "data: [DONE]\n\n"
        return io.BytesIO(wire.encode())

    def test_usage_only_final_event_and_reasoning(self):
        with patch("urllib.request.urlopen", return_value=self.response()):
            result = bench.stream("http://localhost", "test", "prompt", 32, 5)
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["answer"], "42")
        self.assertLessEqual(result["first_delta_s"], result["first_text_s"])

    def test_partial_streams_do_not_score(self):
        for response in (self.response(done=False), self.response(usage=False)):
            with patch("urllib.request.urlopen", return_value=response), self.assertRaises(ValueError):
                bench.stream("http://localhost", "test", "prompt", 32, 5)
        self.assertNotIn("output_tokens_per_s", bench.summary([{"error": "timeout"}], 1))


if __name__ == "__main__":
    unittest.main()
