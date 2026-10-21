# Fixed rules only. Rules derived from the node inventory live in the workers and gpu-access
# stacks. Everything not allowed here or there is denied by the VPC's implied ingress deny.

resource "google_compute_firewall" "cp_iap_ssh" {
  name    = "fw-cp-allow-iap-ssh"
  network = google_compute_network.core.name

  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"] # Google IAP TCP forwarding
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  target_service_accounts = [google_service_account.cp.email]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

# cp reaching its own public IP (hairpin), e.g. a kubeconfig or agent pointing at CP_IP.
resource "google_compute_firewall" "cp_self" {
  name    = "fw-cp-allow-self"
  network = google_compute_network.core.name

  direction     = "INGRESS"
  source_ranges = ["${google_compute_address.cp.address}/32"]
  allow {
    protocol = "tcp"
    ports    = ["6443", "10250"]
  }
  target_service_accounts = [google_service_account.cp.email]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}
