terraform {
  # >= 1.9.0 also lets the gpu_ips validation reference var.max_nodes.
  required_version = ">= 1.9.0"

  # terraform init -backend-config=bucket=<project_id>-k3s-tfstate -backend-config=prefix=gpu-access
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
}
