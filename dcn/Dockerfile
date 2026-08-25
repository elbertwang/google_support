FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1

RUN /opt/venv/bin/python -m venv --clear /opt/dcn-venv \
    && /opt/dcn-venv/bin/pip install --no-cache-dir \
      "jax[tpu]==0.11.0" "libtpu==0.0.44" \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

WORKDIR /workspace/google_support
COPY dcn /workspace/google_support/dcn
ENV DCN_CODE_ROOT=/workspace/google_support \
    DCN_PYTHON=/opt/dcn-venv/bin/python \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["/workspace/google_support/dcn/k8s/entrypoint.sh"]
