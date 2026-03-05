FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg

# Install torch CPU
RUN pip install --no-cache-dir \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install pykan + deps in one shot to avoid conflicts
RUN pip install --no-cache-dir \
    "pykan==0.2.1" \
    "numpy==1.26.4" \
    "pandas==2.1.4" \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    "sympy==1.12" \
    "matplotlib==3.8.2" \
    "tqdm==4.66.1" \
    "requests==2.31.0" \
    "pyyaml>=6.0"

# Debug: show what's actually installed and what fails
#RUN python3 -c "import matplotlib; matplotlib.use('Agg'); print('matplotlib OK')"
#RUN python3 -c "import torch; print('torch', torch.__version__)"
#RUN python3 -c "import sympy; print('sympy OK')"
#RUN python3 -c "import tqdm; print('tqdm OK')"

# Verify pykan import
#RUN python3 -c "import matplotlib; matplotlib.use('Agg'); from kan import KAN; print('pykan OK')"

COPY . .
RUN mkdir -p data solution/models /tmp/kan_traces
RUN python3 datasets/generate.py
RUN chmod +x run.sh

ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV SOLUTION_DIR=/app/solution
ENV TRACE_DIR=/tmp/kan_traces

# CMD is defined in docker-compose.yml per service