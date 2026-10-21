# Per-GPU-node ingress, applied by gpuctl with -auto-approve whenever the Lambda IPs change.
# Every rule is keyed by one GPU IP and allows exactly that /32. An empty gpu_ips list must
# yield no rule at all: GCP opens an ingress rule without a source to everyone (spec 3.2).

data "terraform_remote_state" "base" {
  backend = "gcs"
  config = {
    bucket = var.state_bucket
    prefix = "base"
  }
}

locals {
  base = data.terraform_remote_state.base.outputs
}

# GPU node -> control plane: API server, WireGuard (flannel wireguard-native), ping.
resource "google_compute_firewall" "gpu_to_cp" {
  for_each = toset(var.gpu_ips)

  name    = "fw-gpu-${replace(each.key, ".", "-")}-to-cp"
  network = local.base.core_network

  direction     = "INGRESS"
  source_ranges = ["${each.key}/32"]
  allow {
    protocol = "tcp"
    ports    = ["6443"]
  }
  allow {
    protocol = "udp"
    ports    = ["51820"]
  }
  allow {
    protocol = "icmp"
  }
  target_service_accounts = [local.base.cp_sa_email]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

# GPU node -> CPU workers: WireGuard and ping only.
resource "google_compute_firewall" "gpu_to_workers" {
  for_each = toset(var.gpu_ips)

  name    = "fw-gpu-${replace(each.key, ".", "-")}-to-workers"
  network = local.base.edge_network

  direction     = "INGRESS"
  source_ranges = ["${each.key}/32"]
  allow {
    protocol = "udp"
    ports    = ["51820"]
  }
  allow {
    protocol = "icmp"
  }
  target_service_accounts = [local.base.worker_sa_email]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}
