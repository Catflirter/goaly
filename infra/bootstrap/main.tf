locals {
  apis = toset(["compute", "iap", "oslogin", "iam", "cloudresourcemanager", "storage"])
}

resource "google_project_service" "apis" {
  for_each = local.apis

  project = var.project_id
  service = "${each.key}.googleapis.com"
  # Turning an API off on destroy would break anything else in the project that uses it.
  disable_on_destroy = false
}

# Versioned, private bucket for the remote state of the other stacks (prefix = stack name).
# The gcs backend locks state natively.
resource "google_storage_bucket" "tfstate" {
  name     = "${var.project_id}-k3s-tfstate"
  project  = var.project_id
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}
