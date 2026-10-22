# Rules derived from the worker inventory (spec section 3.1). Every source is a single static
# IP as a /32, except IAP. The per-worker rules use for_each over var.cpu_workers, so a removed
# worker takes its rules with it and no rule is ever generated without a source.

resource "google_compute_firewall" "worker_to_cp" {
  for_each = var.cpu_workers

  name    = "fw-${each.key}-to-cp"
  network = local.base.core_network

  direction     = "INGRESS"
  source_ranges = ["${google_compute_address.worker[each.key].address}/32"]
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

resource "google_compute_firewall" "cp_to_workers" {
  name    = "fw-cp-to-workers"
  network = local.base.edge_network

  direction     = "INGRESS"
  source_ranges = ["${local.base.cp_ip}/32"]
  allow {
    protocol = "tcp"
    ports    = ["10250"]
  }
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

resource "google_compute_firewall" "worker_mesh" {
  for_each = var.cpu_workers

  name    = "fw-${each.key}-mesh"
  network = local.base.edge_network

  direction     = "INGRESS"
  source_ranges = ["${google_compute_address.worker[each.key].address}/32"]
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

resource "google_compute_firewall" "workers_iap_ssh" {
  name    = "fw-workers-allow-iap-ssh"
  network = local.base.edge_network

  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"] # Google IAP TCP forwarding
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  target_service_accounts = [local.base.worker_sa_email]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}
