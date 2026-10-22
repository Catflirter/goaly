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

resource "google_compute_address" "worker" {
  for_each = var.cpu_workers

  name   = "${each.key}-ip"
  region = replace(each.value.zone, "/-[a-z]+$/", "") # us-central1-a -> us-central1
}

resource "google_compute_instance" "worker" {
  for_each = var.cpu_workers

  name                      = each.key
  machine_type              = each.value.machine_type
  zone                      = each.value.zone
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
    subnetwork = local.base.edge_subnetwork
    access_config {
      nat_ip = google_compute_address.worker[each.key].address
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
    email  = local.base.worker_sa_email
    scopes = ["https://www.googleapis.com/auth/userinfo.email"]
  }

  lifecycle {
    # The edge subnet exists only in edge_region; a zone elsewhere would fail mid-apply.
    precondition {
      condition     = startswith(each.value.zone, "${local.base.edge_region}-")
      error_message = "cpu_workers[\"${each.key}\"].zone must be a zone of the edge region ${local.base.edge_region}."
    }
    # A new image in the family must not force a rebuild of the workers.
    ignore_changes = [boot_disk[0].initialize_params[0].image]
  }
}
