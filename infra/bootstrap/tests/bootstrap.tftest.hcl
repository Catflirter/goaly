mock_provider "google" {}
variables { project_id = "demo-project" }

run "state_bucket_is_private_and_versioned" {
  command = apply
  assert {
    condition = (google_storage_bucket.tfstate.name == "demo-project-k3s-tfstate"
      && google_storage_bucket.tfstate.location == "us-west1"
      && google_storage_bucket.tfstate.versioning[0].enabled
      && google_storage_bucket.tfstate.uniform_bucket_level_access
    && google_storage_bucket.tfstate.public_access_prevention == "enforced")
    error_message = "state bucket: <project>-k3s-tfstate, versioned, uniform access, public access prevented"
  }
  assert {
    condition     = output.state_bucket == "demo-project-k3s-tfstate"
    error_message = "state_bucket output"
  }
}

run "required_apis_stay_enabled_on_destroy" {
  command = apply
  assert {
    condition = (toset([for s in google_project_service.apis : s.service]) == toset([
      "compute.googleapis.com", "iap.googleapis.com", "oslogin.googleapis.com", "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com", "storage.googleapis.com"]))
    error_message = "the six required APIs"
  }
  assert {
    condition     = alltrue([for s in google_project_service.apis : !s.disable_on_destroy])
    error_message = "disable_on_destroy must be false"
  }
}
