variable "project_id" {
  description = "GCP project that hosts the cluster."
  type        = string
}

variable "cp_zone" {
  description = "Zone of the k3s-cp control-plane VM."
  type        = string
  default     = "us-west1-b"

  # The static IP and core subnet live in cp_region; a zone elsewhere would fail mid-apply.
  validation {
    condition     = startswith(var.cp_zone, "${var.cp_region}-")
    error_message = "cp_zone must be a zone of cp_region (e.g. us-west1-b for us-west1)."
  }
}

variable "cp_region" {
  description = "Region of the k3s-core VPC subnet and the control plane's static IP."
  type        = string
  default     = "us-west1"
}

variable "edge_region" {
  description = "Region of the k3s-edge VPC subnet (CPU workers)."
  type        = string
  default     = "us-central1"
}

variable "core_cidr" {
  description = "Subnet range of the k3s-core VPC."
  type        = string
  default     = "10.10.0.0/24"
}

variable "edge_cidr" {
  description = "Subnet range of the k3s-edge VPC."
  type        = string
  default     = "10.20.0.0/24"
}

variable "cp_machine_type" {
  description = "Machine type of k3s-cp."
  type        = string
  default     = "e2-medium"
}

variable "boot_image" {
  description = "Boot image (family) for k3s-cp."
  type        = string
  default     = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
}

variable "disk_size_gb" {
  description = "Boot disk size of k3s-cp in GB."
  type        = number
  default     = 20
}
