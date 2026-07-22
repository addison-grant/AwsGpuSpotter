terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Persistent EBS data volume
#
# Lives in a single explicit AZ so it can always reattach to whatever Spot
# instance the ASG launches next, regardless of which instance type in the
# fallback hierarchy actually got capacity. delete_on_termination is handled
# on the *attachment*, not the volume itself -- this resource is never
# destroyed by instance/ASG churn.
# ---------------------------------------------------------------------------
resource "aws_ebs_volume" "persistent_data" {
  availability_zone = var.availability_zone
  size              = var.ebs_volume_size_gb
  type              = "gp3"
  throughput        = 125
  iops              = 3000
  encrypted         = true

  tags = {
    Name    = "${var.project_name}-persistent-data"
    Project = var.project_name
  }

  lifecycle {
    prevent_destroy = true
  }
}

# ---------------------------------------------------------------------------
# Security group
# ---------------------------------------------------------------------------
resource "aws_security_group" "gpu_worker" {
  name        = "${var.project_name}-sg"
  description = "SG for SAM 3.1 GPU spot worker"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  ingress {
    description = "FastAPI / video stream"
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_http_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# ---------------------------------------------------------------------------
# IAM role: allows the instance to discover + attach its own persistent volume
# on boot (see user_data.sh) even after a spot reclaim + replacement launch.
# ---------------------------------------------------------------------------
resource "aws_iam_role" "gpu_worker" {
  name = "${var.project_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "attach_ebs" {
  name = "${var.project_name}-attach-ebs"
  role = aws_iam_role.gpu_worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ec2:AttachVolume",
        "ec2:DescribeVolumes",
        "ec2:DescribeInstances"
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_instance_profile" "gpu_worker" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.gpu_worker.name
}

# ---------------------------------------------------------------------------
# Launch template
# ---------------------------------------------------------------------------
resource "aws_launch_template" "gpu_worker" {
  name_prefix = "${var.project_name}-lt-"
  image_id    = var.ami_id
  key_name    = var.key_name

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu_worker.name
  }

  vpc_security_group_ids = [aws_security_group.gpu_worker.id]

  # Root (ephemeral) volume: OS, NVIDIA drivers, docker images.
  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = var.root_volume_size_gb
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  instance_market_options {
    market_type = "spot"

    spot_options {
      spot_instance_type            = "one-time"
      instance_interruption_behavior = "terminate"
      max_price                     = var.spot_max_price != "" ? var.spot_max_price : null
    }
  }

  metadata_options {
    http_tokens   = "required" # IMDSv2
    http_endpoint = "enabled"
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    ebs_volume_id      = aws_ebs_volume.persistent_data.id
    availability_zone  = var.availability_zone
    app_port           = var.app_port
  }))

  tag_specifications {
    resource_type = "instance"

    tags = {
      Name    = "${var.project_name}-worker"
      Project = var.project_name
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# Auto Scaling Group
#
# Pinned to a single AZ/subnet so the persistent EBS volume (also pinned to
# that AZ) can always reattach to whichever instance the mixed-instances
# fallback hierarchy actually launches.
# ---------------------------------------------------------------------------
resource "aws_autoscaling_group" "gpu_worker" {
  name                = "${var.project_name}-asg"
  vpc_zone_identifier = [var.subnet_id]
  min_size            = 1
  max_size            = 1
  desired_capacity    = 1
  health_check_type   = "EC2"

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                = 0
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                = "price-capacity-optimized"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu_worker.id
        version             = "$Latest"
      }

      # Fallback hierarchy: g6.xlarge (L4) -> g5.xlarge (A10G) -> g4dn.xlarge (T4)
      dynamic "override" {
        for_each = var.instance_types

        content {
          instance_type = override.value
        }
      }
    }
  }

  instance_refresh {
    strategy = "Rolling"

    preferences {
      min_healthy_percentage = 0
    }
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-worker"
    propagate_at_launch = true
  }

  tag {
    key                 = "Project"
    value               = var.project_name
    propagate_at_launch = true
  }
}
