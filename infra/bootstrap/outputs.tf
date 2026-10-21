output "state_bucket" {
  description = "Bucket for the other stacks: terraform init -backend-config=bucket=<this> -backend-config=prefix=<stack>."
  value       = google_storage_bucket.tfstate.name
}
