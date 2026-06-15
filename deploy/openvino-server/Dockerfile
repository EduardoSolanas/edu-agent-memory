FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Install runtime dependencies and Intel OpenCL ICD for iGPU acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 \
    intel-opencl-icd \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install OpenVINO GenAI & FastAPI pipeline packages
RUN pip install --no-cache-dir \
    openvino==2026.2.0 \
    openvino-genai==2026.2.0 \
    fastapi \
    uvicorn \
    pydantic \
    numpy

# Copy server.py to the container workspace
COPY server.py .

EXPOSE 3002

# Run OpenVINO Inference Server
CMD ["python", "server.py"]
