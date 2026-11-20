# RoCE and routed TCP on four switchless Sparks

The lab verified all 12 directed rank-to-rank IPv4 paths over the physical ConnectX ring, including both pairs of opposite nodes. The configuration below gives Gloo and NCCL one stable local IPv4 address per rank. TCP traffic to the opposite rank passes through one other Spark. The two-node RoCE recipes continue to use the direct ConnectX links. PyTorch NCCL also completed an eight-iteration CUDA all-reduce at 1 KiB, 64 KiB and 1 MiB. Every element on every rank contained the expected sum. Across ranks, median latency was 1.00–1.03 ms, 1.43–1.58 ms and 3.80–3.95 ms respectively. Each median excludes the first collective at that size. [Rank-level timings](results/qwen38-27b-scaling-2026-11-19/ring-collectives.jsonl)

## Address the ring

Label nodes A–B–C–D in cable order. On each node identify the ConnectX Ethernet device facing the next node clockwise (`CW`) and the device facing the previous node (`CCW`). Use the following example addresses or substitute an unused internal range. The fabric uses a single Ethernet rail per edge for this TCP baseline.

| Rank | Rank fabric IP | CW link IP | CCW link IP | Next hops to the other three fabric IPs |
| --- | --- | --- | --- | --- |
| A | 198.51.100.1 | 192.0.2.1/30 | 192.0.2.14/30 | B and C via 192.0.2.2, D via 192.0.2.13 |
| B | 198.51.100.2 | 192.0.2.5/30 | 192.0.2.2/30 | A via 192.0.2.1, C and D via 192.0.2.6 |
| C | 198.51.100.3 | 192.0.2.9/30 | 192.0.2.6/30 | A and B via 192.0.2.5, D via 192.0.2.10 |
| D | 198.51.100.4 | 192.0.2.13/30 | 192.0.2.10/30 | A via 192.0.2.14, B and C via 192.0.2.9 |

Each fabric IP lives on a Linux dummy interface. The three host routes select the outgoing physical link and the fabric source address. The following commands show rank A. Substitute its own row on each of the other nodes. These are runtime settings.

```sh
sudo ip addr add 192.0.2.1/30 dev "$CW"
sudo ip addr add 192.0.2.14/30 dev "$CCW"
sudo ip link add fabric0 type dummy
sudo ip addr add 198.51.100.1/32 dev fabric0
sudo ip link set fabric0 up
sudo ip route add 198.51.100.2/32 via 192.0.2.2 dev "$CW" src 198.51.100.1
sudo ip route add 198.51.100.3/32 via 192.0.2.2 dev "$CW" src 198.51.100.1
sudo ip route add 198.51.100.4/32 via 192.0.2.13 dev "$CCW" src 198.51.100.1
```

B forwards A↔C, and C forwards B↔D. On those two nodes, check `sysctl net.ipv4.ip_forward` and the host forwarding rules. The lab's Docker hosts had IPv4 forwarding enabled and a default-drop forwarding policy. The added rules permitted only the two exact rank IPs and the two ConnectX devices on each transit host. For example, on B with Docker's iptables backend:

```sh
sudo iptables -I DOCKER-USER -i "$CCW" -o "$CW" \
  -s 198.51.100.1/32 -d 198.51.100.3/32 -j ACCEPT
sudo iptables -I DOCKER-USER -i "$CW" -o "$CCW" \
  -s 198.51.100.3/32 -d 198.51.100.1/32 -j ACCEPT
```

On C the corresponding source/destination pair is `.2` and `.4`. Use `ping -I "$NODE_IP" -c 1 "$PEER_IP"` for each peer on each rank. For a tensor-parallel trial, run the [CUDA collective probe](README.md#one-endpoint-on-multiple-sparks) first and supply `--transport tcp --interface fabric0` to the launcher. The API on the head remains on loopback. Record this transport separately from switched RoCE or a ring-specific RDMA engine.

## Use RoCE on the same four cables

Keep the routed rank addresses for rendezvous and Gloo. Select the two RoCE HCAs on each rank that face its two neighbors in ring order, one per cable. Each of the four edges has its own /30 IPv4 subnet. The fabric range below selects the corresponding IPv4 GIDs. The lab used one PCI-domain function per neighbor. In the direct two-Spark throughput runs both PCI-domain functions faced the same neighbor.

On the four-node ring, stock NCCL 2.31.2 in the pinned ARM64 vLLM image completed the CUDA collective probe and carried hardware-counted unicast RDMA bytes on both neighbor links of all four ranks. The per-rank median all-reduce times were **0.47–0.58 ms at 1 KiB**, **0.47–0.48 ms at 64 KiB** and **1.425–1.482 ms at 1 MiB**. Timing includes CUDA synchronization and a host round trip after each collective. [Per-port RDMA byte deltas](results/qwen38-27b-scaling-2026-11-19/ring-roce-bytes.csv) · [Per-rank RoCE timings](results/qwen38-27b-scaling-2026-11-19/ring-roce-collectives.jsonl) · [Original TCP comparison](results/qwen38-27b-scaling-2026-11-19/ring-collectives.jsonl)

Run one instance per Spark, substituting the rank number, local rank IP and HCAs. In this example rank zero is A and the four link networks occupy `192.0.2.0/28` as four /30s.

```sh
python3 spark.py serve qwen38-27b-nvfp4 --nodes 4 --rank "$RANK" \
  --head 198.51.100.1 --node-ip "$NODE_IP" --interface fabric0 \
  --transport ring-roce --hcas "$CCW_HCA,$CW_HCA" \
  --fabric-range 192.0.2.0/28 --link-prefix 30
```

The launcher selects NCCL's internal IB transport, ring algorithm, subnet-aware routing, separate NICs, cross-NIC paths and host-staged transfers. Check both neighbor NICs' `rx_vport_rdma_unicast_bytes` and `tx_vport_rdma_unicast_bytes` with `ethtool -S` before and after the probe. Check the log’s rank order and `via NET/IB` lines. Each rank checks eight sums at each message size. [NCCL source for the tested version](https://github.com/NVIDIA/nccl/tree/7b83616df3ae082a1f32bb74c27458bfe8153a13) · [Upstream Spark ring networking](https://github.com/eugr/spark-vllm-docker/blob/ce3257d06ebb8e3552683d59c553bfc64f43d9e7/docs/NETWORKING.md)
