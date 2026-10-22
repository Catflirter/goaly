output "worker_ips" {
  description = "Static public IP of each CPU worker, by node name."
  value       = { for name, a in google_compute_address.worker : name => a.address }
}

output "worker_internal_ips" {
  value = { for name, i in google_compute_instance.worker : name => i.network_interface[0].network_ip }
}

output "worker_zones" {
  value = { for name, i in google_compute_instance.worker : name => i.zone }
}
