variable "project_id" {
  description = "GCP project that hosts the cluster."
  type        = string
}

variable "region" {
  description = "Location of the Terraform state bucket."
  type        = string
  default     = "us-west1"
}
