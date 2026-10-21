"""Shared helpers for the offline Kubernetes manifest policy tests (tests/k8s/)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "k8s"


def load_docs(path) -> list[dict]:
    """Every non-empty YAML document in a (multi-document) manifest file, in file order."""
    with open(path, encoding="utf-8") as f:
        return [d for d in yaml.safe_load_all(f) if d is not None]


def one(docs, kind, name) -> dict:
    """The single document of this kind and metadata.name; fails unless there is exactly one."""
    found = [d for d in docs if d.get("kind") == kind and d.get("metadata", {}).get("name") == name]
    assert len(found) == 1, f"expected exactly one {kind}/{name}, found {len(found)}"
    return found[0]


def pod_spec(doc) -> dict | None:
    """The PodSpec of a Pod, or of a workload that wraps one in spec.template (Deployment, DaemonSet, Job, ...).

    None for objects that carry no Pod (Namespace, Service, NetworkPolicy, ...).
    """
    if doc["kind"] == "Pod":
        return doc["spec"]
    template = doc.get("spec", {}).get("template")
    return template["spec"] if template else None


def all_containers(podspec) -> list[dict]:
    """Init containers followed by regular containers."""
    return [*podspec.get("initContainers", []), *podspec.get("containers", [])]
