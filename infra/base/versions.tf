terraform {
  required_version = ">= 1.9.0"

  # terraform init -backend-config=bucket=<project_id>-k3s-tfstate -backend-config=prefix=base
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.7.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.cp_region
}
