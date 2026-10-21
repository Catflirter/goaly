# Two custom-mode VPCs with no peering: the control plane and the CPU workers only reach
# each other over public /32s, the same path the Lambda GPU nodes use.
resource "google_compute_network" "core" {
  name                    = "k3s-core"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "core" {
  name          = "k3s-core"
  network       = google_compute_network.core.id
  region        = var.cp_region
  ip_cidr_range = var.core_cidr
}

resource "google_compute_network" "edge" {
  name                    = "k3s-edge"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "edge" {
  name          = "k3s-edge"
  network       = google_compute_network.edge.id
  region        = var.edge_region
  ip_cidr_range = var.edge_cidr
}
