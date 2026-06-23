# CServe control plane on cosmos-9 (`services` user)

Production runs as **`services`** via systemd. Development stays in **`/home/praveen/CServe`**.

| | Path |
|---|------|
| Dev (edit, commit, push) | `/home/praveen/CServe` |
| Production (systemd) | `/home/services/CServe` |
| Python venv | `/home/services/cserve-venv` |
| SQLite DB | `/var/lib/cserve/events.db` |
| Unit | `cserve-control.service` |

Both trees share the same git remote (`origin` → GitHub). After you commit on `praveen`:

```bash
/home/praveen/CServe/scripts/sync-cserve-to-services.sh
docker run --rm --privileged --pid=host alpine nsenter -t 1 -m -u -i -n -p systemctl restart cserve-control
```

Or on `services`: `cd ~/CServe && git pull` (uses `~/.git-credentials` on that account).

**Do not** start `/home/praveen/CServe/scripts/run-control-plane.sh` on cosmos-9 (port 8002 conflict).

## Logs (from cosmos-9 dashboard)

| What | Where |
|------|--------|
| Control plane | **Admin → Control plane logs** (journalctl, ~3s refresh) or `GET /dashboard/api/control-plane/logs` |
| vLLM replica | **Model page → replica → vLLM tab** (live tail ~2s from worker `~/.cserve/logs/vllm-<id>.log`) |
| CServe jobs/health | Same model page, **CServe jobs** / **CServe health** tabs (SQLite) |
| Worker agent | `journalctl` / `~/cserve-agent.log` on each GPU node |

Replica live logs require an updated **node agent** on workers (`deploy-agents.sh`).

Head logs: `journalctl -u cserve-control -f`

One-time migration: `scripts/migrate-head-to-services.sh` (already applied 2026-06-02).
