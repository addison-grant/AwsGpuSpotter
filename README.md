# AwsGpuSpotter
Spin up spot GPU instances quickly for dev iteration

Local-to-remote computer vision workflow: live SAM 3.1 video inference running on
an AWS GPU Spot instance, viewed in real time from a browser on your laptop.

## Layout

- `terraform/` — Spot-backed Auto Scaling Group (Launch Template, mixed instances
  policy with a `g6.xlarge` → `g5.xlarge` → `g4dn.xlarge` fallback hierarchy,
  `price-capacity-optimized` allocation), a persistent gp3 EBS volume pinned to a
  single AZ, and `user_data.sh` to reattach/mount it on every (re)launch.
- `backend/` — `uv`-managed FastAPI app.
  - `app/app.py` — loads SAM 3.1 onto CUDA once at startup, serves
    `/video-stream` (MJPEG), and exposes `/start-run`, `/stop-run`, `/dev-reload`.
  - `app/cv_worker.py` — `process_sam_frame(frame, predictor)`; the module hot-
    reloaded by `/dev-reload` without ever touching the loaded model weights.
- `docker-compose.yml` — `nvidia` runtime GPU passthrough, bind-mounts the
  persistent EBS path (`/mnt/sam3-data`) into the container so both app code and
  checkpoints survive Spot interruptions.
- `frontend/` — React + TypeScript + Vite app: embeds the `/video-stream` feed and
  wires the three control buttons to their endpoints.

## Quickstart

```bash
# Infra
cd terraform
cp terraform.tfvars.example terraform.tfvars   # fill in your AMI/VPC/subnet/key
terraform init
terraform apply

# Backend (on the instance, or locally with a GPU)
cd backend
uv sync
uv run uvicorn app.app:app --host 0.0.0.0 --port 8000

# Frontend (local laptop)
cd frontend
npm install
VITE_BACKEND_URL=http://<instance-public-ip>:8000 npm run dev
```

## Hot reload without losing VRAM state

Edit `backend/app/cv_worker.py`, then click **Hot-Reload Python Code** (or
`curl -X POST http://<host>:8000/dev-reload`). This calls `importlib.reload()`
on just that module — the SAM 3.1 predictor loaded in `app.py`'s lifespan
handler is untouched, so multi-GB checkpoints never leave GPU memory.
