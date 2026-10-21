output "gpu_ips" {
  description = "GPU node IPs that currently have firewall rules (sorted, unique)."
  value       = sort(distinct(var.gpu_ips))
}
