FROM python:3.14-slim
WORKDIR /workspace

# Node.js 18 + npm required for Claude Code CLI
RUN apt-get update && apt-get install -y \
    git curl build-essential nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Python deps (baked in for faster container startup)
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ".[v4]"

CMD ["bash"]
