variable "project_id" {
  description = "GCP project that hosts the cluster."
  type        = string
}

variable "state_bucket" {
  description = "GCS bucket holding the Terraform state (<project_id>-k3s-tfstate); the base stack's outputs are read from its prefix base."
  type        = string
}

# The CPU pool inventory. workers.auto.tfvars.json holds its committed value. No default, so a
# missing inventory fails the plan instead of planning the pool away.
variable "cpu_workers" {
  description = "CPU workers by node name (k3s-cpu-<n>): zone and machine type."
  type = map(object({
    zone         = string
    machine_type = string
  }))

  validation {
    condition     = alltrue([for name in keys(var.cpu_workers) : can(regex("^k3s-cpu-[0-9]+$", name))])
    error_message = "Every cpu_workers key must be a node name of the form k3s-cpu-<n>."
  }

  # k3s-cpu-1 is never removed (spec section 10.2), so the pool is never empty either.
  validation {
    condition     = contains(keys(var.cpu_workers), "k3s-cpu-1")
    error_message = "cpu_workers must contain k3s-cpu-1."
  }
}

variable "boot_image" {
  description = "Boot image (family) for the CPU workers."
  type        = string
  default     = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
}

variable "disk_size_gb" {
  description = "Boot disk size of each CPU worker in GB."
  type        = number
  default     = 20
}
