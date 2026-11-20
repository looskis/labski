FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
WORKDIR /app
COPY tabby/ /app/
COPY exllama-requirements.txt /tmp/exllama-requirements.txt
RUN python3 -m pip install --no-cache-dir --no-build-isolation -r /tmp/exllama-requirements.txt . uvloop==0.22.1
