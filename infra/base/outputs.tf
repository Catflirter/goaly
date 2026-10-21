output "cp_name" {
  value = google_compute_instance.cp.name
}

output "cp_zone" {
  value = google_compute_instance.cp.zone
}

output "cp_ip" {
  description = "Static public IP of k3s-cp (CP_IP)."
  value       = google_compute_address.cp.address
}

output "cp_internal_ip" {
  value = google_compute_instance.cp.network_interface[0].network_ip
}

output "core_network" {
  value = google_compute_network.core.name
}

output "edge_network" {
  value = google_compute_network.edge.name
}

output "edge_subnetwork" {
  value = google_compute_subnetwork.edge.id
}

output "edge_region" {
  value = google_compute_subnetwork.edge.region
}

output "cp_sa_email" {
  value = google_service_account.cp.email
}

output "worker_sa_email" {
  value = google_service_account.worker.email
}
