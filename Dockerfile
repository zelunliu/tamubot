FROM python:3.14-slim
WORKDIR /workspace

# System deps + Claude Code CLI + Bun
RUN apt-get update && apt-get install -y \
    git curl build-essential nodejs npm unzip \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code \
    && curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash

# Python deps (baked in for faster container startup)
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ".[v4]"

# Non-root user (required for claude --dangerously-skip-permissions)
RUN useradd -m -s /bin/bash claude && chown -R claude:claude /workspace

# Bake ccstatusline config so it persists across container restarts
COPY .ccstatusline-settings.json /home/claude/.config/ccstatusline/settings.json
RUN chown -R claude:claude /home/claude/.config

USER claude

CMD ["bash"]
