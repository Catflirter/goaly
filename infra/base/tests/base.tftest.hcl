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
  assert {
    condition     = google_compute_instance.cp.allow_stopping_for_update
    error_message = "allow_stopping_for_update must be true"
  }
  assert {
    condition = (length(google_compute_instance.cp.service_account) == 1
      && google_compute_instance.cp.service_account[0].email == google_service_account.cp.email
    && toset(google_compute_instance.cp.service_account[0].scopes) == toset(["https://www.googleapis.com/auth/userinfo.email"]))
    error_message = "cp runs as the cp service account with only the userinfo.email scope"
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
    condition = (length(google_compute_firewall.cp_self.allow) == 1
      && alltrue([for a in google_compute_firewall.cp_self.allow :
    a.protocol == "tcp" && toset(a.ports) == toset(["6443", "10250"])]))
    error_message = "cp_self opens 6443 and 10250 only"
  }
  assert {
    condition = (length(google_compute_firewall.cp_iap_ssh.allow) == 1
      && alltrue([for a in google_compute_firewall.cp_iap_ssh.allow :
    a.protocol == "tcp" && toset(a.ports) == toset(["22"])]))
    error_message = "cp_iap_ssh opens tcp 22 only"
  }
  assert {
    condition     = google_compute_firewall.cp_iap_ssh.name == "fw-cp-allow-iap-ssh" && google_compute_firewall.cp_self.name == "fw-cp-allow-self"
    error_message = "firewall rule names"
  }
  assert {
    condition = (google_compute_firewall.cp_iap_ssh.direction == "INGRESS"
      && toset(google_compute_firewall.cp_iap_ssh.target_service_accounts) == toset([google_service_account.cp.email])
    && try(length(google_compute_firewall.cp_iap_ssh.target_tags), 0) == 0)
    error_message = "cp_iap_ssh: ingress, targets only the cp service account, no tags"
  }
  assert {
    condition     = length(google_compute_firewall.cp_iap_ssh.log_config) == 1 && google_compute_firewall.cp_iap_ssh.log_config[0].metadata == "INCLUDE_ALL_METADATA"
    error_message = "cp_iap_ssh: log_config with INCLUDE_ALL_METADATA"
  }
  assert {
    condition = (google_compute_firewall.cp_self.direction == "INGRESS"
      && toset(google_compute_firewall.cp_self.target_service_accounts) == toset([google_service_account.cp.email])
    && try(length(google_compute_firewall.cp_self.target_tags), 0) == 0)
    error_message = "cp_self: ingress, targets only the cp service account, no tags"
  }
  assert {
    condition     = length(google_compute_firewall.cp_self.log_config) == 1 && google_compute_firewall.cp_self.log_config[0].metadata == "INCLUDE_ALL_METADATA"
    error_message = "cp_self: log_config with INCLUDE_ALL_METADATA"
  }
}

run "outputs" {
  command = apply
  assert {
    condition     = output.cp_ip == "34.82.10.10" && output.cp_zone == "us-west1-b" && output.edge_region == "us-central1"
    error_message = "outputs"
  }
}

run "cp_zone_must_be_in_cp_region" {
  command = plan
  variables {
    cp_zone = "us-east1-b"
  }
  expect_failures = [var.cp_zone]
}
