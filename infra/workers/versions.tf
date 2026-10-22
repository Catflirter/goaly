terraform {
  required_version = ">= 1.9.0"

  # terraform init -backend-config=bucket=<project_id>-k3s-tfstate -backend-config=prefix=workers
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.7.0"
    }
  }
}

# Every resource sets its own zone or region (derived from var.cpu_workers), or is global.
provider "google" {
  project = var.project_id
}
