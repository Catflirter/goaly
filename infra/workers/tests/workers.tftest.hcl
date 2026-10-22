mock_provider "google" {}

variables {
  project_id   = "demo"
  state_bucket = "demo-k3s-tfstate"
}

override_data {
  target = data.terraform_remote_state.base
  values = {
    outputs = {
      core_network    = "k3s-core"
      edge_network    = "k3s-edge"
      edge_subnetwork = "projects/demo/regions/us-central1/subnetworks/k3s-edge"
      edge_region     = "us-central1"
      cp_ip           = "34.82.10.10"
      cp_sa_email     = "k3s-cp@demo.iam.gserviceaccount.com"
      worker_sa_email = "k3s-worker@demo.iam.gserviceaccount.com"
    }
  }
}

override_resource {
  target = google_compute_address.worker
  values = { address = "34.29.20.20" }
}

# No cpu_workers here: the committed inventory (workers.auto.tfvars.json) must load and validate.
run "committed_inventory_plans" {
  command = plan
  assert {
    condition     = contains(keys(google_compute_instance.worker), "k3s-cpu-1")
    error_message = "the committed inventory holds k3s-cpu-1"
  }
}

run "one_worker" {
  command = apply
  variables {
    cpu_workers = { "k3s-cpu-1" = { zone = "us-central1-a", machine_type = "e2-small" } }
  }

  assert {
    condition = (length(google_compute_instance.worker) == 1
      && google_compute_instance.worker["k3s-cpu-1"].machine_type == "e2-small"
    && google_compute_instance.worker["k3s-cpu-1"].zone == "us-central1-a")
    error_message = "one e2-small worker in us-central1-a"
  }
  assert {
    condition = (google_compute_address.worker["k3s-cpu-1"].name == "k3s-cpu-1-ip"
    && google_compute_address.worker["k3s-cpu-1"].region == "us-central1")
    error_message = "the worker's static IP is named <name>-ip and lives in the zone's region"
  }
  assert {
    condition = (google_compute_instance.worker["k3s-cpu-1"].network_interface[0].subnetwork == "projects/demo/regions/us-central1/subnetworks/k3s-edge"
    && google_compute_instance.worker["k3s-cpu-1"].network_interface[0].access_config[0].nat_ip == "34.29.20.20")
    error_message = "the worker sits in the edge subnet behind its static IP"
  }
  assert {
    condition = alltrue([google_compute_instance.worker["k3s-cpu-1"].shielded_instance_config[0].enable_secure_boot,
      google_compute_instance.worker["k3s-cpu-1"].shielded_instance_config[0].enable_vtpm,
    google_compute_instance.worker["k3s-cpu-1"].shielded_instance_config[0].enable_integrity_monitoring])
    error_message = "all three Shielded VM options must be true"
  }
  assert {
    condition = (google_compute_instance.worker["k3s-cpu-1"].metadata["enable-oslogin"] == "TRUE"
      && google_compute_instance.worker["k3s-cpu-1"].metadata["block-project-ssh-keys"] == "TRUE"
    && google_compute_instance.worker["k3s-cpu-1"].can_ip_forward == false)
    error_message = "OS Login on, project SSH keys blocked, no ip forwarding"
  }
  assert {
    condition = (google_compute_instance.worker["k3s-cpu-1"].boot_disk[0].initialize_params[0].image == "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      && google_compute_instance.worker["k3s-cpu-1"].boot_disk[0].initialize_params[0].size == 20
    && google_compute_instance.worker["k3s-cpu-1"].boot_disk[0].initialize_params[0].type == "pd-balanced")
    error_message = "boot disk"
  }
  assert {
    condition = (length(google_compute_instance.worker["k3s-cpu-1"].service_account) == 1
      && google_compute_instance.worker["k3s-cpu-1"].service_account[0].email == "k3s-worker@demo.iam.gserviceaccount.com"
    && toset(google_compute_instance.worker["k3s-cpu-1"].service_account[0].scopes) == toset(["https://www.googleapis.com/auth/userinfo.email"]))
    error_message = "the worker runs as the worker service account with only the userinfo.email scope"
  }

  assert {
    condition = (google_compute_firewall.worker_to_cp["k3s-cpu-1"].source_ranges == toset(["34.29.20.20/32"])
      && google_compute_firewall.worker_to_cp["k3s-cpu-1"].network == "k3s-core"
      && google_compute_firewall.worker_to_cp["k3s-cpu-1"].name == "fw-k3s-cpu-1-to-cp"
    && toset(google_compute_firewall.worker_to_cp["k3s-cpu-1"].target_service_accounts) == toset(["k3s-cp@demo.iam.gserviceaccount.com"]))
    error_message = "worker_to_cp: only the worker's /32, on k3s-core, targeting the cp service account"
  }
  assert {
    condition = toset([for a in google_compute_firewall.worker_to_cp["k3s-cpu-1"].allow :
    "${a.protocol}:${join(",", a.ports == null ? [] : a.ports)}"]) == toset(["tcp:6443", "udp:51820", "icmp:"])
    error_message = "worker_to_cp opens tcp 6443, udp 51820 and icmp only"
  }
  assert {
    condition = (google_compute_firewall.cp_to_workers.source_ranges == toset(["34.82.10.10/32"])
      && google_compute_firewall.cp_to_workers.network == "k3s-edge"
      && google_compute_firewall.cp_to_workers.name == "fw-cp-to-workers"
    && toset(google_compute_firewall.cp_to_workers.target_service_accounts) == toset(["k3s-worker@demo.iam.gserviceaccount.com"]))
    error_message = "cp_to_workers: only cp's /32, on k3s-edge, targeting the worker service account"
  }
  assert {
    condition = toset([for a in google_compute_firewall.cp_to_workers.allow :
    "${a.protocol}:${join(",", a.ports == null ? [] : a.ports)}"]) == toset(["tcp:10250", "udp:51820", "icmp:"])
    error_message = "cp_to_workers opens tcp 10250, udp 51820 and icmp only"
  }
  assert {
    condition = (google_compute_firewall.worker_mesh["k3s-cpu-1"].source_ranges == toset(["34.29.20.20/32"])
      && google_compute_firewall.worker_mesh["k3s-cpu-1"].network == "k3s-edge"
      && google_compute_firewall.worker_mesh["k3s-cpu-1"].name == "fw-k3s-cpu-1-mesh"
    && toset(google_compute_firewall.worker_mesh["k3s-cpu-1"].target_service_accounts) == toset(["k3s-worker@demo.iam.gserviceaccount.com"]))
    error_message = "worker_mesh: only the worker's /32, on k3s-edge, targeting the worker service account"
  }
  assert {
    condition = toset([for a in google_compute_firewall.worker_mesh["k3s-cpu-1"].allow :
    "${a.protocol}:${join(",", a.ports == null ? [] : a.ports)}"]) == toset(["udp:51820", "icmp:"])
    error_message = "worker_mesh opens udp 51820 and icmp only"
  }
  assert {
    condition = (google_compute_firewall.workers_iap_ssh.source_ranges == toset(["35.235.240.0/20"])
      && google_compute_firewall.workers_iap_ssh.network == "k3s-edge"
      && google_compute_firewall.workers_iap_ssh.name == "fw-workers-allow-iap-ssh"
      && toset(google_compute_firewall.workers_iap_ssh.target_service_accounts) == toset(["k3s-worker@demo.iam.gserviceaccount.com"])
      && toset([for a in google_compute_firewall.workers_iap_ssh.allow :
    "${a.protocol}:${join(",", a.ports == null ? [] : a.ports)}"]) == toset(["tcp:22"]))
    error_message = "workers_iap_ssh: tcp 22 from IAP only, on k3s-edge, targeting the worker service account"
  }
  assert {
    condition = alltrue([for r in concat(values(google_compute_firewall.worker_to_cp), values(google_compute_firewall.worker_mesh),
      [google_compute_firewall.cp_to_workers, google_compute_firewall.workers_iap_ssh]) :
    r.direction == "INGRESS" && length(r.log_config) == 1 && r.log_config[0].metadata == "INCLUDE_ALL_METADATA"])
    error_message = "every rule is ingress with log_config INCLUDE_ALL_METADATA"
  }

  assert {
    condition = (output.worker_ips == { "k3s-cpu-1" = "34.29.20.20" }
      && output.worker_zones == { "k3s-cpu-1" = "us-central1-a" }
    && keys(output.worker_internal_ips) == ["k3s-cpu-1"])
    error_message = "outputs map each worker name to its external IP, internal IP and zone"
  }
}

run "scale_to_three" {
  command = plan
  variables {
    cpu_workers = {
      "k3s-cpu-1" = { zone = "us-central1-a", machine_type = "e2-small" }
      "k3s-cpu-2" = { zone = "us-central1-a", machine_type = "e2-small" }
      "k3s-cpu-3" = { zone = "us-central1-b", machine_type = "e2-standard-2" }
    }
  }

  assert {
    condition     = length(google_compute_instance.worker) == 3 && length(google_compute_address.worker) == 3
    error_message = "three workers, three static IPs"
  }
  assert {
    condition     = length(google_compute_firewall.worker_to_cp) == 3 && length(google_compute_firewall.worker_mesh) == 3
    error_message = "one worker_to_cp and one worker_mesh rule per worker"
  }
  assert {
    condition     = google_compute_firewall.worker_mesh["k3s-cpu-3"].name == "fw-k3s-cpu-3-mesh" && google_compute_firewall.worker_to_cp["k3s-cpu-2"].name == "fw-k3s-cpu-2-to-cp"
    error_message = "per-worker rule names"
  }
}

run "k3s_cpu_1_is_mandatory" {
  command = plan
  variables {
    cpu_workers = { "k3s-cpu-2" = { zone = "us-central1-a", machine_type = "e2-small" } }
  }
  expect_failures = [var.cpu_workers]
}

run "bad_name_rejected" {
  command = plan
  variables {
    cpu_workers = {
      "k3s-cpu-1" = { zone = "us-central1-a", machine_type = "e2-small" }
      "cpu2"      = { zone = "us-central1-a", machine_type = "e2-small" }
    }
  }
  expect_failures = [var.cpu_workers]
}

run "zone_must_be_in_edge_region" {
  command = plan
  variables {
    cpu_workers = { "k3s-cpu-1" = { zone = "us-west1-b", machine_type = "e2-small" } }
  }
  expect_failures = [google_compute_instance.worker]
}
