# Docker Setup — TAMUBot Dev Environment

Two paths to a running sandbox: **Windows 11** (via WSL2 + WSLg) and **Mac**.

---

## Windows 11 — WSL2 + WSLg

### 1. Verify WSLg (Windows 11 22H2+)

```bash
wsl --version
# Requires: WSL version 2.x.x, Kernel 5.15+
```

If WSL is not installed: `wsl --install` then restart.

### 2. Install Docker Engine in WSL2

Inside your WSL2 Ubuntu/Debian shell:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, then verify:
docker run hello-world
```

### 3. Install Ghostty inside WSL2

Ghostty is not in Debian/Ubuntu apt repos — install via `.deb`:

```bash
# Download the latest .deb from https://github.com/ghostty-org/ghostty/releases
# Example (check for latest version):
wget https://github.com/ghostty-org/ghostty/releases/download/v1.x.x/ghostty_1.x.x_amd64.deb
sudo dpkg -i ghostty_*.deb
```

WSLg renders the Ghostty window on your Windows desktop automatically — no Xserver needed.

### 4. Migrate project to WSL2 filesystem

Exclude `.venv` and `__pycache__` (Windows venv won't run on Linux):

```bash
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  /mnt/c/dev/TAMU_NEW/ ~/dev/TAMU_NEW/
```

### 5. Fix git + SSH inside WSL2

```bash
cd ~/dev/TAMU_NEW
git remote set-url origin git@github.com:<org>/TAMU_NEW.git

# Copy SSH keys from Windows (or generate new ones)
cp /mnt/c/Users/$WINDOWS_USER/.ssh/id_ed25519 ~/.ssh/
chmod 600 ~/.ssh/id_ed25519
ssh -T git@github.com   # verify
```

### 6. Re-create Python venv

```bash
cd ~/dev/TAMU_NEW
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[v4]"
```

### 7. Copy `.env`

```bash
cp /mnt/c/dev/TAMU_NEW/.env ~/dev/TAMU_NEW/.env
```

### 8. Start the sandbox

```bash
make sandbox-up
```

---

## Mac

### 1. Install Docker Desktop

Download from https://www.docker.com/products/docker-desktop — install and start it.

### 2. Install Ghostty (native)

Download from https://ghostty.org — drag to `/Applications`.

### 3. Clone / pull the project

```bash
git clone git@github.com:<org>/TAMU_NEW.git ~/dev/TAMU_NEW
cd ~/dev/TAMU_NEW
cp .env.example .env   # fill in API keys
```

### 4. Start the sandbox

```bash
make sandbox-up
```

---

## Common — After `make sandbox-up`

### Verify containers are running

```bash
docker ps
# Should show: tamubot-claude-1, tamubot-api-proxy-1, tamubot-app-1
```

### Open a Claude Code session

```bash
make sandbox-shell
# Inside container:
claude --dangerously-skip-permissions
```

### VS Code Dev Containers

1. Install extension: **Dev Containers** (`ms-vscode-remote.remote-containers`)
2. Open the project folder in VS Code
3. Command palette → **Dev Containers: Reopen in Container**
4. VS Code attaches to the `claude` service automatically

### Streamlit app

http://localhost:8501

### Tear down

```bash
make sandbox-down
```

---

## Troubleshooting

**`${HOME}` not expanded in docker-compose.yml**
Make sure you're running `docker compose` (not `docker-compose` v1). V2 expands `${HOME}`.

**Port 8501 already in use**
Kill the conflicting process: `lsof -ti:8501 | xargs kill`

**`tamubot-claude-1` container name not found**
Run `docker ps --format '{{.Names}}'` to see actual container names, then update `sandbox-shell` in Makefile if needed.

**WSLg window not appearing**
Verify `DISPLAY` is set: `echo $DISPLAY` should show `:0` or similar. Requires Windows 11 build 22000+.
