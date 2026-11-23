"""Deterministic, leak-free multi-hop serial reasoning tasks + real LM/math datasets.

Design principles:
  1. Disjoint entity/label vocabularies between demo shots, train split, and eval split
     (zero entity memorization possible).
  2. Two clean in-context format demonstrations in every prompt so even the untrained
     1-pass base model produces the exact output format immediately.
  3. Both in-distribution (ID) and deeper out-of-distribution (OOD) hop/step counts
     to directly test whether Inkling 2's recurrent loop adds real serial depth.
  4. Real natural-language data (WikiText-2 + GSM8K) mixed into training as an LM
     preservation regularizer and evaluated on strictly held-out test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any


# Disjoint symbol pools (demo vs train vs eval)
DEMO_KEYS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH"]
TRAIN_KEYS = [
    f"{a}{b}"
    for a in "BCDFGHJKLM"
    for b in "123456789"
]  # 90 keys
EVAL_KEYS = [
    f"{a}{b}"
    for a in "NPQRSTVWXZ"
    for b in "123456789"
]  # 90 completely disjoint keys

DEMO_ITEMS = ["apple", "brick", "chair", "drum", "eagle"]
TRAIN_ITEMS = [
    "amber", "badge", "cabin", "delta", "ember", "flint", "grain", "haven",
    "ivory", "joker", "knoll", "lemon", "maple", "noble", "olive", "pearl",
    "quilt", "river", "slate", "tiger", "umbra", "vigor", "wheat", "xenon",
]
EVAL_ITEMS = [
    "acorn", "blaze", "coral", "drift", "epoch", "frost", "gleam", "heron",
    "inlet", "jewel", "kayak", "lotus", "mango", "nexus", "otter", "plume",
    "quartz", "ridge", "spark", "tulip", "ultra", "vault", "waltz", "yacht",
]


@dataclass
class TaskExample:
    task: str
    split: str          # "train", "eval_id", "eval_ood"
    depth: int          # hops / steps / distance / swaps
    prompt: str
    answer: str
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Task 1: Multi-Hop Pointer Chasing (chained key-value lookups, no CoT)
# ---------------------------------------------------------------------------

_POINTER_DEMO = (
    "Follow the exact number of pointer hops in the table and reply with only the final key.\n\n"
    "Table: DD->EE, BB->CC, AA->BB, CC->DD, EE->FF\n"
    "Start: AA | Hops: 2 | Final key: CC\n\n"
    "Table: HH->AA, FF->GG, GG->HH, EE->FF\n"
    "Start: EE | Hops: 3 | Final key: HH\n\n"
)


def make_pointer_chase(rng: random.Random, split: str, hops: int, n_nodes: int = 12) -> TaskExample:
    pool = TRAIN_KEYS if split == "train" else EVAL_KEYS
    nodes = rng.sample(pool, n_nodes)
    # Build a random permutation (disjoint cycles) or single long cycle so every hop is well-defined
    chain = list(nodes)
    rng.shuffle(chain)
    mapping = {chain[i]: chain[(i + 1) % n_nodes] for i in range(n_nodes)}
    start = rng.choice(chain)
    cur = start
    for _ in range(hops):
        cur = mapping[cur]
    answer = cur

    entries = [f"{k}->{v}" for k, v in mapping.items()]
    rng.shuffle(entries)
    prompt = (
        _POINTER_DEMO
        + f"Table: {', '.join(entries)}\n"
        + f"Start: {start} | Hops: {hops} | Final key:"
    )
    return TaskExample(
        task="pointer_chase",
        split=split,
        depth=hops,
        prompt=prompt,
        answer=" " + answer,
        meta={"start": start, "hops": hops, "target": answer, "n_nodes": n_nodes},
    )


# ---------------------------------------------------------------------------
# Task 2: Sequential Modular Register Execution (mod 19, no CoT)
# ---------------------------------------------------------------------------

_MOD_DEMO = (
    "Compute the final value of r modulo 19 after executing all steps in order. Reply with only the integer.\n\n"
    "Init: r = 4 | Steps: r = (r + 5) mod 19; r = (r * 3) mod 19 | Final r: 8\n\n"
    "Init: r = 10 | Steps: r = (r - 3) mod 19; r = (r * 2) mod 19; r = (r + 6) mod 19 | Final r: 1\n\n"
)


def make_mod_register(rng: random.Random, split: str, steps: int, mod: int = 19) -> TaskExample:
    r = rng.randint(1, mod - 1)
    init_r = r
    ops_str = []
    for _ in range(steps):
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            c = rng.randint(2, 6)
            r = (r * c) % mod
        elif op == "+":
            c = rng.randint(1, 9)
            r = (r + c) % mod
        else:
            c = rng.randint(1, 9)
            r = (r - c) % mod
        ops_str.append(f"r = (r {op} {c}) mod {mod}")
    prompt = (
        _MOD_DEMO
        + f"Init: r = {init_r} | Steps: {'; '.join(ops_str)} | Final r:"
    )
    return TaskExample(
        task="mod_register",
        split=split,
        depth=steps,
        prompt=prompt,
        answer=" " + str(r),
        meta={"init_r": init_r, "steps": steps, "target": str(r)},
    )


# ---------------------------------------------------------------------------
# Task 3: Directed Graph Shortest Path / BFS (inspired by GraphWalks BFS)
# ---------------------------------------------------------------------------

_GRAPH_DEMO = (
    "Find the shortest directed path length from Start to Goal in the graph, or NONE if unreachable. Reply with only the integer or NONE.\n\n"
    "Edges: CC->DD, AA->BB, BB->CC, AA->EE, EE->FF\n"
    "Query: AA to DD | Shortest path: 3\n\n"
    "Edges: BB->CC, AA->BB, DD->EE, EE->FF\n"
    "Query: AA to EE | Shortest path: NONE\n\n"
)


def _bfs_dist(adj: dict[str, list[str]], src: str, dst: str) -> int | None:
    seen = {src: 0}
    q = [src]
    for u in q:
        d = seen[u]
        if u == dst:
            return d
        for v in adj.get(u, []):
            if v not in seen:
                seen[v] = d + 1
                q.append(v)
    return None


def make_graph_bfs(rng: random.Random, split: str, target_dist: int, n_nodes: int = 11) -> TaskExample:
    """Generate a sparse directed graph with exact shortest path = target_dist (0 means NONE)."""
    pool = TRAIN_KEYS if split == "train" else EVAL_KEYS
    for _attempt in range(200):
        nodes = rng.sample(pool, n_nodes)
        adj: dict[str, list[str]] = {u: [] for u in nodes}
        if target_dist == 0:
            # Partition into 2 disconnected components so src -> dst is unreachable
            cut = n_nodes // 2
            comp_a, comp_b = nodes[:cut], nodes[cut:]
            for comp in (comp_a, comp_b):
                for i in range(len(comp) - 1):
                    adj[comp[i]].append(comp[i + 1])
                for _ in range(len(comp)):
                    u, v = rng.sample(comp, 2)
                    if v not in adj[u]:
                        adj[u].append(v)
            # Add reverse edges from B -> A only (still unreachable A -> B!)
            for _ in range(2):
                u = rng.choice(comp_b)
                v = rng.choice(comp_a)
                if v not in adj[u]:
                    adj[u].append(v)
            src, dst = comp_a[0], comp_b[-1]
            if _bfs_dist(adj, src, dst) is None:
                ans = "NONE"
                break
        else:
            # Plant a backbone path of exact length target_dist
            path = nodes[: target_dist + 1]
            for i in range(target_dist):
                adj[path[i]].append(path[i + 1])
            # Add distractor edges that do NOT shorten path[0] -> path[-1]
            distractors = nodes[target_dist + 1 :]
            all_pairs = [(u, v) for u in nodes for v in nodes if u != v]
            rng.shuffle(all_pairs)
            added = 0
            for u, v in all_pairs:
                if added >= n_nodes + 1:
                    break
                if v in adj[u]:
                    continue
                adj[u].append(v)
                d = _bfs_dist(adj, path[0], path[-1])
                if d != target_dist:
                    adj[u].pop()
                else:
                    added += 1
            src, dst = path[0], path[-1]
            if _bfs_dist(adj, src, dst) == target_dist:
                ans = str(target_dist)
                break
    else:
        raise RuntimeError(f"Failed to construct graph_bfs for dist={target_dist}")

    edges = [f"{u}->{v}" for u, vs in adj.items() for v in vs]
    rng.shuffle(edges)
    prompt = (
        _GRAPH_DEMO
        + f"Edges: {', '.join(edges)}\n"
        + f"Query: {src} to {dst} | Shortest path:"
    )
    return TaskExample(
        task="graph_bfs",
        split=split,
        depth=target_dist,
        prompt=prompt,
        answer=" " + ans,
        meta={"src": src, "dst": dst, "target": ans, "n_edges": len(edges)},
    )


# ---------------------------------------------------------------------------
# Task 4: S_5 Permutation / Cup-Swap State Tracking (NC^1 vs TC^0 separator)
# ---------------------------------------------------------------------------

_PERM_DEMO = (
    "Track the items in slots 1 through 5 after performing all swaps in order. Reply with only the item name.\n\n"
    "Slots: 1=apple, 2=brick, 3=chair, 4=drum, 5=eagle | Swaps: swap(1,3), swap(3,5) | Slot 5: apple\n\n"
    "Slots: 1=apple, 2=brick, 3=chair, 4=drum, 5=eagle | Swaps: swap(2,4), swap(1,2), swap(4,5) | Slot 2: apple\n\n"
)


def make_perm_track(rng: random.Random, split: str, n_swaps: int) -> TaskExample:
    pool = TRAIN_ITEMS if split == "train" else EVAL_ITEMS
    items = rng.sample(pool, 5)
    state = list(items)
    swaps = []
    prev_pair = None
    for _ in range(n_swaps):
        while True:
            i, j = sorted(rng.sample(range(5), 2))
            if (i, j) != prev_pair:
                break
        prev_pair = (i, j)
        state[i], state[j] = state[j], state[i]
        swaps.append(f"swap({i + 1},{j + 1})")
    query_slot = rng.randint(0, 4)
    answer = state[query_slot]
    init_str = ", ".join(f"{k + 1}={v}" for k, v in enumerate(items))
    prompt = (
        _PERM_DEMO
        + f"Slots: {init_str} | Swaps: {', '.join(swaps)} | Slot {query_slot + 1}:"
    )
    return TaskExample(
        task="perm_track",
        split=split,
        depth=n_swaps,
        prompt=prompt,
        answer=" " + answer,
        meta={"n_swaps": n_swaps, "query_slot": query_slot + 1, "target": answer},
    )


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def build_serial_train_examples(seed: int = 1001, per_task: int = 600) -> list[TaskExample]:
    rng = random.Random(seed)
    examples: list[TaskExample] = []
    for _ in range(per_task):
        examples.append(make_pointer_chase(rng, "train", hops=rng.choice([2, 3, 4])))
        examples.append(make_mod_register(rng, "train", steps=rng.choice([2, 3, 4])))
        examples.append(make_graph_bfs(rng, "train", target_dist=rng.choice([0, 1, 2, 3])))
        examples.append(make_perm_track(rng, "train", n_swaps=rng.choice([3, 4, 5, 6])))
    rng.shuffle(examples)
    return examples


def build_serial_eval_suite(seed: int = 20261122, per_depth: int = 24) -> list[TaskExample]:
    """Deterministic held-out eval suite covering both ID and OOD depths across all 4 tasks."""
    rng = random.Random(seed)
    examples: list[TaskExample] = []

    # Pointer chasing: ID={2,3,4}, OOD={5,6}
    for hops in (2, 3, 4):
        for _ in range(per_depth):
            examples.append(make_pointer_chase(rng, "eval_id", hops=hops))
    for hops in (5, 6):
        for _ in range(per_depth):
            examples.append(make_pointer_chase(rng, "eval_ood", hops=hops))

    # Modular register: ID={2,3,4}, OOD={5,6}
    for steps in (2, 3, 4):
        for _ in range(per_depth):
            examples.append(make_mod_register(rng, "eval_id", steps=steps))
    for steps in (5, 6):
        for _ in range(per_depth):
            examples.append(make_mod_register(rng, "eval_ood", steps=steps))

    # Graph BFS: ID={0(NONE), 2, 3}, OOD={4, 5}
    for d in (0, 2, 3):
        for _ in range(per_depth):
            examples.append(make_graph_bfs(rng, "eval_id", target_dist=d))
    for d in (4, 5):
        for _ in range(per_depth):
            examples.append(make_graph_bfs(rng, "eval_ood", target_dist=d))

    # Permutation tracking: ID={3,4,5,6}, OOD={7,8}
    for swaps in (3, 4, 5, 6):
        for _ in range(per_depth // 2 + 6):
            examples.append(make_perm_track(rng, "eval_id", n_swaps=swaps))
    for swaps in (7, 8):
        for _ in range(per_depth):
            examples.append(make_perm_track(rng, "eval_ood", n_swaps=swaps))

    return examples


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_GSM8K_DEMO = (
    "Solve the math word problem and give only the final numeric answer.\n\n"
    "Question: A store sells 4 boxes of pencils with 12 pencils in each box. How many pencils are there in total?\n"
    "Answer: 48\n\n"
    "Question: Maya has 35 marbles. She gives 9 to Leo and buys 14 more. How many marbles does Maya have now?\n"
    "Answer: 40\n\n"
)


def extract_gsm8k_gold(answer_text: str) -> str:
    m = re.search(r"####\s*([-+]?[0-9,]+(?:\.[0-9]+)?)", answer_text)
    if not m:
        return answer_text.strip()
    return m.group(1).replace(",", "").strip()


def format_gsm8k_direct(row: dict[str, Any]) -> tuple[str, str]:
    gold = extract_gsm8k_gold(row["answer"])
    prompt = f"{_GSM8K_DEMO}Question: {row['question'].strip()}\nAnswer:"
    return prompt, " " + gold


def normalize_answer_str(text: str) -> str:
    s = text.strip()
    # Stop at first newline or punctuation delimiter
    s = re.split(r"[\n\r|.;,]", s)[0].strip()
    # Strip trailing period/comma
    return s.rstrip(".,;:!?").strip()
