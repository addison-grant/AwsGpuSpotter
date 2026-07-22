variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "Single explicit AZ to bind the ASG + EBS volume to (avoids cross-AZ fees and guarantees reattachment)"
  type        = string
  default     = "us-east-1a"
}

variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "sam3-gpu-spotter"
}

variable "instance_types" {
  description = "Ordered fallback hierarchy of GPU instance types: L4 -> A10G -> T4"
  type        = list(string)
  default     = ["g6.xlarge", "g5.xlarge", "g4dn.xlarge"]
}

variable "ami_id" {
  description = "Deep Learning / NVIDIA GPU-optimized AMI ID (Amazon Linux 2023 or Ubuntu DLAMI) for the chosen region"
  type        = string
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "ebs_volume_size_gb" {
  description = "Size in GB of the persistent gp3 data volume (model weights + code + checkpoints)"
  type        = number
  default     = 100

  validation {
    condition     = var.ebs_volume_size_gb >= 100
    error_message = "The persistent EBS volume must be at least 100GB to hold SAM 3.1 checkpoints and working data."
  }
}

variable "root_volume_size_gb" {
  description = "Size in GB of the ephemeral root volume (OS + drivers + docker images)"
  type        = number
  default     = 100
}

variable "allowed_ssh_cidr" {
  description = "CIDR block permitted to SSH into the instance"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_http_cidr" {
  description = "CIDR block permitted to reach the FastAPI app / video stream"
  type        = string
  default     = "0.0.0.0/0"
}

variable "app_port" {
  description = "Port the FastAPI server listens on"
  type        = number
  default     = 8000
}

variable "spot_max_price" {
  description = "Optional max spot price per hour (empty string = on-demand price cap, recommended for price-capacity-optimized)"
  type        = string
  default     = ""
}

variable "vpc_id" {
  description = "VPC to deploy into"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID in the chosen availability_zone"
  type        = string
}
