"""Launch one native DeepSeek V4.1 Flash 1017 rank on a four-Spark switchless RoCE ring."""
import argparse
import ipaddress
import os
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

LOGICAL_NODES = (0, 1, 3, 2)
# Upstream GB10 round costs in milliseconds for 1 through 48 verified rows.
ROUND_MS = (
    "21.5,26.3,31.1,36.0,40.3,44.7,49.0,53.4,"
    "57.1,60.8,64.5,68.2,72.0,75.7,79.4,83.1,"
    "86.4,89.7,93.0,96.3,99.6,102.9,106.2,109.5,"
    "112.8,116.1,119.4,122.7,126.0,129.3,132.6,135.9,"
    "138.6,141.3,144.0,146.7,149.4,152.1,154.8,157.5,"
    "160.2,162.9,165.6,168.3,171.0,173.7,176.4,179.1"
)


def graphs(directory, rank, previous, following):
    directory.mkdir(parents=True, exist_ok=True)
    for name, member, device in (("world",rank,previous), ("prev",1,previous), ("next",0,following)):
        root = ET.Element("graphs",version="1")
        for index, pattern, active in ((0,4,True),(1,3,True),(4,6,False),(2,3,False),(3,5,False)):
            count = 2 if active else 0
            graph = ET.SubElement(root,"graph",id=str(index),pattern=str(pattern),crossnic=str(int(active)),
                                  nchannels=str(count),speedintra="12" if active else "0",speedinter="12" if active else "0",
                                  latencyinter="0",typeintra="LOC",typeinter="P2C" if active else "LOC",samechannels="1" if active else "0")
            for _ in range(count):
                ch = ET.SubElement(graph,"channel")
                ET.SubElement(ch,"net",dev=str(device))
                ET.SubElement(ch,"gpu",dev="0",rank=str(member))
                ET.SubElement(ch,"net",dev=str(device))
        ET.indent(root)
        ET.ElementTree(root).write(directory/(name+".xml"),encoding="utf-8")


def command(a):
    rank = a.rank
    logical = LOGICAL_NODES[rank]
    head = str(ipaddress.IPv4Address(a.head))
    fabric_range = str(ipaddress.IPv4Network(a.address_range,strict=True))
    if len(a.hcas.split(","))!=2 or not all(a.hcas.split(",")):
        raise ValueError("--hcas must name the two NCCL devices in physical order")
    weights = list(a.rank_cache.glob(f"rank{logical}of4-*.bin"))
    if len(weights)!=1 or weights[0].stat().st_size==0:
        raise ValueError(f"expected one prepared rank{logical}of4 weight file")
    for item in (a.model/"config.json", a.common/"dsv41_token_map.json", a.common/"kit"/"MANIFEST", a.ring_nccl):
        if not item.is_file():
            raise ValueError(f"missing required rank asset {item}")
    if len(list((a.common/"engram").glob("*.safetensors")))!=2:
        raise ValueError("expected both original Engram checkpoint shards")
    graphs(a.graph_dir,rank,a.previous_device,a.next_device)
    name="deepseek1017-native"
    cmd=["docker","run","--rm","--pull=never","--name","labski-"+name,"--gpus","all","--ipc","host","--network","host",
         "--ulimit","memlock=-1","--ulimit","stack=67108864","--cap-add","IPC_LOCK","--device","/dev/infiniband",
         "--memory","112g","--memory-swap","112g"]
    for path,target in ((a.model,"/model"),(a.common/"kit","/kit"),(a.common/"engram","/engram"),
                        (a.common/"dsv41_token_map.json","/token-map.json"),(a.rank_cache,"/rank-cache"),
                        (a.ring_nccl,"/labski/libnccl.so.2"),(a.graph_dir,"/graphs")):
        cmd.extend(["-v",str(path.resolve())+":"+target+":ro"])
    pool=a.context*a.slots
    env=dict(HF_HUB_OFFLINE=1,NCCL_GRAPH_FILE="/graphs/world.xml",TF_DS_LINK_GRAPHS="/graphs/prev.xml,/graphs/next.xml",
             NCCL_SOCKET_IFNAME="="+a.interface,NCCL_IB_HCA="="+a.hcas,NCCL_NET="IB",NCCL_ALGO="Ring",NCCL_NET_PLUGIN="none",
             NCCL_IB_SUBNET_AWARE_ROUTING=1,NCCL_IB_SUBNET_PREFIX_LEN=30,NCCL_IB_ADDR_FAMILY="AF_INET",NCCL_IB_ADDR_RANGE=fabric_range,
             NCCL_IB_MERGE_NICS=0,NCCL_CROSS_NIC=1,NCCL_NET_GDR_LEVEL="LOC",NCCL_CUMEM_ENABLE=0,NCCL_SWITCHLESS_RING_ONLY=1,
             NCCL_P2P_DISABLE=1,NCCL_SHM_DISABLE=1,NCCL_MIN_NCHANNELS=2,NCCL_MAX_NCHANNELS=2,NCCL_DEBUG="INFO",
             NCCL_DEBUG_SUBSYS="INIT,NET,GRAPH",LD_PRELOAD="/labski/libnccl.so.2",TF_DS_2D_GU="parity",TF_DS_2D_ORDER="0,1,3,2",
             TF_DS_ROUND_ROWS=48,TF_DS_STREAMS=a.slots,TF_DS_SERVED_K="solo",TF_DS_WINDOW=pool,TF_DS_GRAPHS=1,
             TF_DS_HC_SIDE=1,TF_DS_L2_PREFETCH=1,TF_DS_ENGRAM_AIO=1,TF_DS_DECODE_DEC=0,TF_DS_ROUND_MS=ROUND_MS,TF_DS_DRAFT_MS=25)
    if rank==0:
        env.update(TF_TP_WORLD=4,TF_TP_MASTER=head,TF_TP_LISTEN=",".join([head]*3),TF_TP_PORT=a.master_port,
                   TF_DS_RANK_CACHE="/rank-cache",TF_DS_KIT="/kit",TENSORFOLD_CUDA_KERNELS="/kit/aot",
                   TF_DS_ENGRAM="/engram",TF_DS_TOKEN_MAP="/token-map.json")
        argv=["tensorfold-native","serve","/model","--backend","cuda","--host","127.0.0.1","--port",str(a.port),"--name",name,
              "--context",str(a.context),"--parallel",str(a.slots),"--max-tokens","512","--temperature","0","--top-p","1",
              "--top-k","1","--no-thinking","--no-update-check"]
    else:
        argv=["tf-dsv41-lanes","/model","/rank-cache",str(rank),"4",head,str(a.master_port),"/kit","--pool",str(pool),
              "--context",str(a.context),"--engram","/engram","--token-map","/token-map.json","--drafts","1","--round-rows","48",
              "--streams",str(a.slots),"--graphs","1","--side","1","--prefetch","1","--aio","1"]
    for key,val in env.items():
        cmd.extend(["-e",f"{key}={val}"])
    return cmd+["--entrypoint",argv[0],a.image,*argv[1:]]


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank",required=True,type=int,choices=range(4))
    p.add_argument("--head",required=True,help="head IPv4 address reachable over the routed ring")
    p.add_argument("--address-range",required=True,help="CIDR covering the point-to-point RoCE links")
    p.add_argument("--interface",required=True)
    p.add_argument("--hcas",required=True)
    for opt in ("model","common","rank-cache","ring-nccl","graph-dir"):
        p.add_argument("--"+opt,required=True,type=Path)
    p.add_argument("--image",default="labski-deepseek1017:8ffa67d-arm64")
    p.add_argument("--previous-device",type=int,default=0)
    p.add_argument("--next-device",type=int,default=1)
    p.add_argument("--context",type=int,default=8192)
    p.add_argument("--slots",type=int,default=4)
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--master-port",type=int,default=29917)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if min(a.context,a.slots)<1 or not all(0<n<65536 for n in (a.port,a.master_port)):
        p.error("context, slot count and ports must be positive and ports below 65536")
    cmd=command(a)
    print(shlex.join(cmd),flush=True)
    if not a.dry_run:os.execvp(cmd[0],cmd)
