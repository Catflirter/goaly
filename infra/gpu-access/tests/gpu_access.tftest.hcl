mock_provider "google" {}
override_data {
  target = data.terraform_remote_state.base
  values = { outputs = { core_network = "k3s-core", edge_network = "k3s-edge",
  cp_sa_email = "k3s-cp@demo.iam.gserviceaccount.com", worker_sa_email = "k3s-worker@demo.iam.gserviceaccount.com" } }
}
variables {
  project_id   = "demo"
  state_bucket = "demo-k3s-tfstate"
}

run "empty_list_creates_no_rules" { # Review Focus #4
  command = apply
  variables { gpu_ips = [] }
  assert {
    condition     = length(google_compute_firewall.gpu_to_cp) == 0 && length(google_compute_firewall.gpu_to_workers) == 0
    error_message = "an empty list must not leave any rule behind (empty source = everyone on GCP)"
  }
  assert {
    condition     = length(output.gpu_ips) == 0
    error_message = "gpu_ips output must be empty"
  }
}

run "default_is_empty" {
  command = apply
  assert {
    condition     = length(google_compute_firewall.gpu_to_cp) == 0 && length(google_compute_firewall.gpu_to_workers) == 0
    error_message = "with no gpu_ips given, no rule may exist"
  }
}

run "one_ip_exact_sources" {
  command = apply
  variables { gpu_ips = ["129.80.30.30"] }
  assert {
    condition = (google_compute_firewall.gpu_to_cp["129.80.30.30"].source_ranges == toset(["129.80.30.30/32"])
    && google_compute_firewall.gpu_to_workers["129.80.30.30"].source_ranges == toset(["129.80.30.30/32"]))
    error_message = "sources must be the single /32"
  }
  assert {
    condition     = google_compute_firewall.gpu_to_cp["129.80.30.30"].name == "fw-gpu-129-80-30-30-to-cp"
    error_message = "rule naming"
  }
  assert {
    condition     = google_compute_firewall.gpu_to_workers["129.80.30.30"].name == "fw-gpu-129-80-30-30-to-workers"
    error_message = "workers rule naming"
  }
  assert {
    condition = (google_compute_firewall.gpu_to_cp["129.80.30.30"].network == "k3s-core"
      && google_compute_firewall.gpu_to_cp["129.80.30.30"].direction == "INGRESS"
    && toset(google_compute_firewall.gpu_to_cp["129.80.30.30"].target_service_accounts) == toset(["k3s-cp@demo.iam.gserviceaccount.com"]))
    error_message = "gpu_to_cp: ingress on k3s-core, targets only the cp service account"
  }
  assert {
    condition = (google_compute_firewall.gpu_to_workers["129.80.30.30"].network == "k3s-edge"
      && google_compute_firewall.gpu_to_workers["129.80.30.30"].direction == "INGRESS"
    && toset(google_compute_firewall.gpu_to_workers["129.80.30.30"].target_service_accounts) == toset(["k3s-worker@demo.iam.gserviceaccount.com"]))
    error_message = "gpu_to_workers: ingress on k3s-edge, targets only the worker service account"
  }
  assert {
    condition = (toset([for a in google_compute_firewall.gpu_to_cp["129.80.30.30"].allow : "${a.protocol}:${a.ports == null ? "" : join(",", sort(a.ports))}"])
    == toset(["tcp:6443", "udp:51820", "icmp:"]))
    error_message = "gpu_to_cp opens tcp 6443, udp 51820 and icmp only"
  }
  assert {
    condition = (toset([for a in google_compute_firewall.gpu_to_workers["129.80.30.30"].allow : "${a.protocol}:${a.ports == null ? "" : join(",", sort(a.ports))}"])
    == toset(["udp:51820", "icmp:"]))
    error_message = "gpu_to_workers opens udp 51820 and icmp only"
  }
  assert {
    condition = (google_compute_firewall.gpu_to_cp["129.80.30.30"].log_config[0].metadata == "INCLUDE_ALL_METADATA"
    && google_compute_firewall.gpu_to_workers["129.80.30.30"].log_config[0].metadata == "INCLUDE_ALL_METADATA")
    error_message = "both rules log with INCLUDE_ALL_METADATA"
  }
}

run "output_is_sorted_and_distinct" {
  command = apply
  variables { gpu_ips = ["150.136.9.9", "129.80.30.30", "150.136.9.9"] }
  assert {
    condition     = output.gpu_ips == tolist(["129.80.30.30", "150.136.9.9"])
    error_message = "gpu_ips output must be sort(distinct(var.gpu_ips))"
  }
  assert {
    condition     = length(google_compute_firewall.gpu_to_cp) == 2 && length(google_compute_firewall.gpu_to_workers) == 2
    error_message = "one rule pair per distinct IP"
  }
}

# Review Focus #4 for real: the last GPU goes offline and gpu_ips drops from N IPs back to [].
# The previous run left two rule pairs in state; this apply must remove every one of them.
run "back_to_empty_removes_all_rules" {
  command = apply
  variables { gpu_ips = [] }
  assert {
    condition     = length(google_compute_firewall.gpu_to_cp) == 0 && length(google_compute_firewall.gpu_to_workers) == 0
    error_message = "dropping gpu_ips back to [] must remove every rule an earlier apply created"
  }
  assert {
    condition     = length(output.gpu_ips) == 0
    error_message = "gpu_ips output must be empty once every GPU is gone"
  }
}

run "max_nodes_ips_accepted" {
  command = plan
  variables { gpu_ips = ["129.80.30.30", "150.136.9.9", "198.51.100.99"] }
  assert {
    condition     = length(google_compute_firewall.gpu_to_cp) == 3
    error_message = "exactly max_nodes IPs must be accepted"
  }
}

# Public addresses right next to each excluded range must pass the prefix checks.
run "neighbours_of_excluded_ranges_accepted" {
  command = plan
  variables {
    max_nodes = 9
    gpu_ips = ["1.0.0.1", "11.0.0.1", "100.63.255.255", "100.128.0.1", "126.255.255.255",
    "172.15.255.255", "172.32.0.1", "192.167.255.255", "192.169.0.1"]
  }
  assert {
    condition     = length(google_compute_firewall.gpu_to_cp) == 9
    error_message = "public neighbours of the excluded ranges must be accepted"
  }
}

run "rejects_cidr" {
  command = plan
  variables { gpu_ips = ["198.51.100.0/24"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_host_cidr" {
  command = plan
  variables { gpu_ips = ["1.2.3.4/32"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_three_octets" {
  command = plan
  variables { gpu_ips = ["129.80.30"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_five_octets" {
  command = plan
  variables { gpu_ips = ["129.80.30.30.1"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_this_network" {
  command = plan
  variables { gpu_ips = ["0.0.0.0"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_rfc1918_10" {
  command = plan
  variables { gpu_ips = ["10.0.0.5"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_rfc1918_172" {
  command = plan
  variables { gpu_ips = ["172.16.4.4"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_rfc1918_172_top" {
  command = plan
  variables { gpu_ips = ["172.31.255.255"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_rfc1918_192" {
  command = plan
  variables { gpu_ips = ["192.168.1.1"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_cgnat" {
  command = plan
  variables { gpu_ips = ["100.64.0.1"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_cgnat_top" {
  command = plan
  variables { gpu_ips = ["100.127.255.255"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_loopback" {
  command = plan
  variables { gpu_ips = ["127.0.0.1"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_octet_over_255" {
  command = plan
  variables { gpu_ips = ["300.1.1.1"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_empty_string" {
  command = plan
  variables { gpu_ips = [""] }
  expect_failures = [var.gpu_ips]
}

# A zero-padded octet would slip past the prefix checks ("010.0.0.5" does not start with "10.").
run "rejects_leading_zero" {
  command = plan
  variables { gpu_ips = ["010.0.0.5"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_bad_entry_among_good" {
  command = plan
  variables { gpu_ips = ["129.80.30.30", "10.0.0.5"] }
  expect_failures = [var.gpu_ips]
}

run "rejects_more_than_max_nodes" {
  command = plan
  variables { gpu_ips = ["129.80.30.30", "150.136.9.9", "198.51.100.99", "203.0.113.7"] }
  expect_failures = [var.gpu_ips]
}
