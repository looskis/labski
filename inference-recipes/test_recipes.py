"""Offline regression checks for rank construction and streaming measurement."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bench
import mmlu_probe
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
                        if r.get("env", {}).get("B12X_DISK_BACKEND") == "io_uring":
                            profile = spark.ROOT / r["seccomp_profile"]
                            self.assertEqual(cmd[cmd.index("--security-opt") + 1], f"seccomp={profile}")
                            self.assertEqual(json.loads(profile.read_bytes())["defaultAction"], "SCMP_ACT_ERRNO")
                        if r.get("engine") == "tensorfold-glm53":
                            self.assertIn("RANK=" + str(rank), cmd)
                            self.assertIn("PARALLEL=4", cmd)
                            self.assertIn("DRAFTER=mtp", cmd)
                            self.assertIn("MASTER=192.0.2.1:29501", cmd)
                            self.assertIn("TF_GLM53_ROCE=0", cmd)
                            self.assertIn("--host 127.0.0.1", cmd[-1])
                            args.output_head = "bf16"
                            self.assertIn("HEAD_Q=off", spark.serve_command(args, r, Path("/tmp/test cache")))
                            model_path = spark.CONTAINER_CACHE / spark.snapshot(r)
                            self.assertIn(f"MODEL={model_path}", cmd)
                            self.assertEqual(cmd[cmd.index("--volume") + 1], "/tmp/test cache:/weights:ro")
                            args.output_head = "q6"
                            self.assertIn(f"HEAD_Q={model_path}/lm_head-6bit/lm_head-6bit.safetensors",
                                          spark.serve_command(args, r, Path("/tmp/test cache")))
                            continue
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

    def test_switchless_roce_ring(self):
        name = "qwen38-27b-nvfp4"
        recipe = json.loads(spark.RECIPES[name].read_text())
        args = SimpleNamespace(recipe=name, nodes=4, rank=2, port=8000, master_port=29501,
                               head="198.51.100.1", node_ip="198.51.100.3", interface="fabric0",
                               transport="ring-roce", hcas="rdma0,rdma1",
                               fabric_range="192.0.2.0/28", link_prefix=30)
        cmd = spark.serve_command(args, recipe, Path("/tmp/cache"))
        for value in ("NCCL_NET=IB", "NCCL_ALGO=Ring", "NCCL_IB_ADDR_RANGE=192.0.2.0/28",
                      "NCCL_IB_SUBNET_PREFIX_LEN=30", "NCCL_IB_HCA==rdma0,rdma1",
                      "NCCL_IB_SUBNET_AWARE_ROUTING=1", "NCCL_IB_MERGE_NICS=0", "/dev/infiniband"):
            self.assertIn(value, cmd)
        for change in ({"hcas": "rdma0,rdma0"}, {"link_prefix": 27}, {"fabric_range": None},
                       {"nodes": 2, "rank": 1}, {"transport": "roce"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                spark.serve_command(SimpleNamespace(**{**vars(args), **change}), recipe, Path("/tmp/cache"))

    def test_sglang_nccl_override(self):
        name = "inkling2-small-native"
        recipe = json.loads(spark.RECIPES[name].read_text())
        with TemporaryDirectory() as folder:
            library = Path(folder) / "libnccl.so.2"
            library.touch()
            args = SimpleNamespace(recipe=name, nodes=4, rank=0, port=8000, master_port=29501,
                                   head="198.51.100.1", node_ip="198.51.100.1", interface="fabric0",
                                   transport="ring-roce", hcas="rdma0,rdma1", fabric_range="192.0.2.0/28",
                                   ring_nccl=library, ring_only_init=False)
            cmd = spark.serve_command(args, recipe, Path(folder))
            self.assertIn(f"{library}:/labski/libnccl.so.2:ro", cmd)
            self.assertIn("SGLANG_NCCL_SO_PATH=/labski/libnccl.so.2", cmd)
            self.assertNotIn("NCCL_SWITCHLESS_RING_ONLY=1", cmd)
            args.ring_only_init = True
            self.assertIn("NCCL_SWITCHLESS_RING_ONLY=1", spark.serve_command(args, recipe, Path(folder)))
            args.ring_nccl = None
            with self.assertRaises(ValueError):
                spark.serve_command(args, recipe, Path(folder))


class NativeChatTests(unittest.TestCase):
    def test_constrained_token_preserves_rendered_answer(self):
        args = SimpleNamespace(model="example", url="http://localhost", one_letter_token=True)
        allowed = list(range(20, 30))
        choice = dict(token_ids=[27], message={"content": " H"}, finish_reason="length")
        data = dict(choices=[choice], usage={"prompt_tokens": 12, "completion_tokens": 1})
        with patch("mmlu_probe.post", return_value=data) as request:
            row = mmlu_probe.one_letter(args, "Choose a letter", allowed)
            self.assertTrue(request.call_args.args[1]["return_token_ids"])
            self.assertEqual((row["answer"], row["visible_answer"], row["answer_token_id"]), ("H", " H", 27))
            choice["token_ids"] = [91]
            with self.assertRaises(ValueError):
                mmlu_probe.one_letter(args, "Choose a letter", allowed)

    def test_reasoning_only_response_and_output_budget(self):
        args = SimpleNamespace(model="example", url="http://localhost", max_tokens=8, reasoning_effort="low")
        data = {"choices": [{"message": {"content": None, "reasoning_content": "analysis"},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 8}}
        with patch("mmlu_probe.post", return_value=data) as request:
            result = mmlu_probe.native_chat(args, "Choose a letter")
            body = request.call_args.args[1]
            self.assertEqual(body["chat_template_kwargs"], {"reasoning_effort": "low"})
            self.assertEqual(body["max_tokens"], 8)
            self.assertEqual((result["answer"], result["finish"], result["reasoning_chars"]), ("", "length", 8))
            args.reasoning_effort = None
            args.enable_thinking = "off"
            result = mmlu_probe.native_chat(args, "Choose a letter")
            self.assertIs(result["enable_thinking"], False)
            self.assertIs(request.call_args.args[1]["enable_thinking"], False)
            self.assertNotIn("chat_template_kwargs", request.call_args.args[1])
            data["usage"]["completion_tokens"] = 9
            with self.assertRaises(ValueError):
                mmlu_probe.native_chat(args, "Choose a letter")


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
