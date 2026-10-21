# Local state on purpose: this stack creates the bucket every other stack keeps its state in.
terraform {
  required_version = ">= 1.9.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.7.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
