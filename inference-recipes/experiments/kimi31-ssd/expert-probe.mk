# Read after Argodrive's Makefile using make -f Makefile -f /probe/expert-probe.mk.
expert-probe: $(CORE_OBJS) ds4_gpu_args.o
	$(CC) -O2 -I. -c /probe/expert_probe.c -o /probe/expert_probe.o
	$(NVCC) $(NVCC_ARCH_FLAGS) -o /probe/expert-probe /probe/expert_probe.o ds4_gpu_args.o $(CORE_OBJS) -lcublas -Xcompiler -pthread -lm
