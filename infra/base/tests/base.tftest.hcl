mock_provider "google" {}
variables { project_id = "demo-project" }
override_resource {
  target = google_compute_address.cp
  values = { address = "34.82.10.10" }
}

run "cp_instance_is_hardened" {
  command = apply
  assert {
    condition     = google_compute_instance.cp.machine_type == "e2-medium" && google_compute_instance.cp.zone == "us-west1-b"
    error_message = "cp must be e2-medium in us-west1-b"
  }
  assert {
    condition = alltrue([google_compute_instance.cp.shielded_instance_config[0].enable_secure_boot,
      google_compute_instance.cp.shielded_instance_config[0].enable_vtpm,
    google_compute_instance.cp.shielded_instance_config[0].enable_integrity_monitoring])
    error_message = "all three Shielded VM options must be true"
  }
  assert {
    condition     = google_compute_instance.cp.metadata["enable-oslogin"] == "TRUE" && google_compute_instance.cp.metadata["block-project-ssh-keys"] == "TRUE"
    error_message = "OS Login on, project SSH keys blocked"
  }
  assert {
    condition     = google_compute_instance.cp.can_ip_forward == false
    error_message = "no ip forwarding"
  }
  assert {
    condition = (google_compute_instance.cp.boot_disk[0].initialize_params[0].image == "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      && google_compute_instance.cp.boot_disk[0].initialize_params[0].size == 20
    && google_compute_instance.cp.boot_disk[0].initialize_params[0].type == "pd-balanced")
    error_message = "boot disk"
  }
}

run "networks_and_subnets" {
  command = apply
  assert {
    condition = (!google_compute_network.core.auto_create_subnetworks && google_compute_network.core.mtu == 1460
      && !google_compute_network.edge.auto_create_subnetworks && google_compute_network.edge.mtu == 1460
      && google_compute_subnetwork.core.ip_cidr_range == "10.10.0.0/24"
    && google_compute_subnetwork.edge.ip_cidr_range == "10.20.0.0/24" && google_compute_subnetwork.edge.region == "us-central1")
    error_message = "custom-mode VPCs with the spec CIDRs"
  }
}

run "fixed_firewall_rules" {
  command = apply
  assert {
    condition     = google_compute_firewall.cp_iap_ssh.source_ranges == toset(["35.235.240.0/20"])
    error_message = "SSH only from IAP"
  }
  assert {
    condition     = google_compute_firewall.cp_self.source_ranges == toset(["34.82.10.10/32"])
    error_message = "hairpin rule allows only cp's own /32"
  }
  assert {
    condition = anytrue([for a in google_compute_firewall.cp_self.allow :
    a.protocol == "tcp" && toset(a.ports) == toset(["6443", "10250"])])
    error_message = "cp_self opens 6443 and 10250 only"
  }
}

run "outputs" {
  command = apply
  assert {
    condition     = output.cp_ip == "34.82.10.10" && output.cp_zone == "us-west1-b" && output.edge_region == "us-central1"
    error_message = "outputs"
  }
}
