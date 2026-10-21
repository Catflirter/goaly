variable "project_id" {
  description = "GCP project that hosts the cluster."
  type        = string
}

variable "state_bucket" {
  description = "GCS bucket holding the Terraform state (<project_id>-k3s-tfstate); base is read from prefix base."
  type        = string
}

# Second copy of the GPU node cap. It must equal max_nodes in lambda/gpu-nodes.yaml, which is
# the source of truth: if the YAML allows more nodes than this default, gpuctl launches them but
# this stack refuses their IPs and none of them can join. Callers should pass
# -var=max_nodes=<spec max_nodes>; tests/test_terraform_static.py fails while the two differ.
variable "max_nodes" {
  description = "Upper bound on the number of GPU IPs (spec 3.2); must equal max_nodes in lambda/gpu-nodes.yaml."
  type        = number
  default     = 3
}

variable "gpu_ips" {
  description = "Public IPv4 address of every GPU node; each one gets its own /32 rules."
  type        = list(string)
  default     = []

  # Spec 3.2 rule 2: single IPv4 addresses only; the /32 is added by the rules themselves.
  validation {
    condition     = alltrue([for ip in var.gpu_ips : can(regex("^(\\d{1,3}\\.){3}\\d{1,3}$", ip))])
    error_message = "Every gpu_ips entry must be a single IPv4 address such as 129.80.30.30 (no CIDR, no /32, no empty string)."
  }

  # Octets are 0-255 and written without leading zeros: "010.0.0.5" would otherwise slip past
  # the prefix checks below while a parser may still read it as 10.0.0.5 (or octal 8.0.0.5).
  validation {
    condition = alltrue([for ip in var.gpu_ips : alltrue([for o in split(".", ip) :
    can(regex("^(0|[1-9]\\d{0,2})$", o)) && try(tonumber(o) <= 255, false)])])
    error_message = "Every gpu_ips octet must be 0-255 without leading zeros."
  }

  # Prefix checks for 0/8, 10/8, 100.64/10, 127/8, 172.16/12 and 192.168/16: a GPU node is
  # reached over its public address, never over a private, CGNAT or loopback one.
  validation {
    condition = !anytrue([for ip in var.gpu_ips : can(regex(
    "^(0|10|127)\\.|^100\\.(6[4-9]|[7-9]\\d|1[01]\\d|12[0-7])\\.|^172\\.(1[6-9]|2\\d|3[01])\\.|^192\\.168\\.", ip))])
    error_message = "gpu_ips must not contain addresses in 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8, 172.16.0.0/12 or 192.168.0.0/16."
  }

  validation {
    condition     = length(var.gpu_ips) <= var.max_nodes
    error_message = "gpu_ips holds more entries than max_nodes allows."
  }
}
