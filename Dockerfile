# sliceagent — runnable container image.
#   docker build -t sliceagent .
#   docker run -it -e LLM_API_KEY=$LLM_API_KEY -v "$PWD:/work" -w /work sliceagent
# The agent edits files in the mounted workspace (/work); pass your key via -e (never bake it in).
FROM python:3.12-slim

# git: memem is a git dependency · ripgrep: powers the code-index / search tier
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[tui]"

# Drop privileges; the workspace is bind-mounted at runtime.
RUN useradd -m agent
USER agent
WORKDIR /work

ENTRYPOINT ["sliceagent"]
