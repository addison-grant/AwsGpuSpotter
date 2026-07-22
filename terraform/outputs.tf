output "ebs_volume_id" {
  description = "ID of the persistent gp3 data volume (survives spot interruptions)"
  value       = aws_ebs_volume.persistent_data.id
}

output "autoscaling_group_name" {
  value = aws_autoscaling_group.gpu_worker.name
}

output "launch_template_id" {
  value = aws_launch_template.gpu_worker.id
}

output "security_group_id" {
  value = aws_security_group.gpu_worker.id
}

output "app_url_hint" {
  description = "Once the instance is running, find its public IP and browse to this port"
  value       = "http://<instance-public-ip>:${var.app_port}"
}
