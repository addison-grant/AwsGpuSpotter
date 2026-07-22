#!/bin/bash
# Boots on every Spot launch (including replacement after interruption).
# Finds the persistent EBS data volume by ID, attaches it if not already
# attached, mounts it, and brings up the docker-compose stack.
set -euo pipefail

EBS_VOLUME_ID="${ebs_volume_id}"
AVAILABILITY_ZONE="${availability_zone}"
APP_PORT="${app_port}"
DATA_DEVICE="/dev/sdf"
MOUNT_POINT="/mnt/sam3-data"

INSTANCE_ID=$(curl -sf -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')" http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -sf -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')" http://169.254.169.254/latest/meta-data/placement/region)

# Attach the persistent volume if it isn't already attached to this instance.
CURRENT_STATE=$(aws ec2 describe-volumes --region "$REGION" --volume-ids "$EBS_VOLUME_ID" --query 'Volumes[0].State' --output text)

if [ "$CURRENT_STATE" = "available" ]; then
  aws ec2 attach-volume \
    --region "$REGION" \
    --volume-id "$EBS_VOLUME_ID" \
    --instance-id "$INSTANCE_ID" \
    --device "$DATA_DEVICE"

  # Wait for the attachment to become live.
  for i in $(seq 1 30); do
    STATE=$(aws ec2 describe-volumes --region "$REGION" --volume-ids "$EBS_VOLUME_ID" --query 'Volumes[0].Attachments[0].State' --output text)
    [ "$STATE" = "attached" ] && break
    sleep 5
  done
fi

# Wait for the block device to show up in the OS, then format it once (only
# on first-ever boot; a real filesystem signature short-circuits this).
for i in $(seq 1 30); do
  [ -e "$DATA_DEVICE" ] && break
  sleep 2
done

if ! blkid "$DATA_DEVICE" >/dev/null 2>&1; then
  mkfs -t ext4 "$DATA_DEVICE"
fi

mkdir -p "$MOUNT_POINT"
mount "$DATA_DEVICE" "$MOUNT_POINT"
grep -q "$DATA_DEVICE" /etc/fstab || echo "$DATA_DEVICE $MOUNT_POINT ext4 defaults,nofail 0 2" >> /etc/fstab

mkdir -p "$MOUNT_POINT/checkpoints" "$MOUNT_POINT/app"

# Install docker + nvidia-container-toolkit if this AMI doesn't already have it.
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# Application code + docker-compose.yml are expected to already live on the
# persistent volume (rsync'd/deployed separately). If present, bring the
# stack up so the app survives interruption -> replacement with zero manual steps.
if [ -f "$MOUNT_POINT/app/docker-compose.yml" ]; then
  cd "$MOUNT_POINT/app"
  docker compose up -d
fi
