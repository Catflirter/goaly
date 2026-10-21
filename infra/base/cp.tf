# Dedicated service accounts with no IAM role bindings (spec section 4.5).
resource "google_service_account" "cp" {
  account_id   = "k3s-cp"
  display_name = "k3s control plane (no roles)"
}

resource "google_service_account" "worker" {
  account_id   = "k3s-worker"
  display_name = "k3s CPU workers (no roles)"
}

resource "google_compute_address" "cp" {
  name   = "k3s-cp"
  region = var.cp_region
}

resource "google_compute_instance" "cp" {
  name                      = "k3s-cp"
  machine_type              = var.cp_machine_type
  zone                      = var.cp_zone
  can_ip_forward            = false
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = var.boot_image
      size  = var.disk_size_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.core.id
    access_config {
      nat_ip = google_compute_address.cp.address
    }
  }

  # secure boot defaults to false in the provider, so all three are set explicitly.
  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  metadata = {
    enable-oslogin         = "TRUE"
    block-project-ssh-keys = "TRUE"
  }

  service_account {
    email  = google_service_account.cp.email
    scopes = ["https://www.googleapis.com/auth/userinfo.email"]
  }

  lifecycle {
    # A new image in the family must not force a rebuild of the control plane.
    ignore_changes = [boot_disk[0].initialize_params[0].image]
  }
}
