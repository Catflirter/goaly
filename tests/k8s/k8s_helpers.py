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


# Kinds whose spec.template is a PodTemplateSpec.
TEMPLATE_KINDS = frozenset({"Deployment", "DaemonSet", "StatefulSet", "ReplicaSet", "ReplicationController", "Job"})
# Kinds that never carry a Pod. Any other kind makes pod_spec() fail, so a new workload kind (or a typo)
# cannot slip past the image-pinning and Pod Security walks by being treated as Pod-less.
PODLESS_KINDS = frozenset({
    "Namespace", "Service", "ServiceAccount", "ConfigMap", "Secret", "NetworkPolicy", "RuntimeClass",
    "Role", "RoleBinding", "ClusterRole", "ClusterRoleBinding", "APIService", "PodDisruptionBudget",
    "PriorityClass", "ResourceQuota", "LimitRange",
})


def pod_spec(doc) -> dict | None:
    """The PodSpec of a Pod, of a workload that wraps one in spec.template (Deployment, DaemonSet, Job, ...),
    or of a CronJob (spec.jobTemplate.spec.template).

    None only for the explicitly Pod-less kinds (Namespace, Service, NetworkPolicy, ...); any other kind raises.
    """
    kind = doc["kind"]
    if kind == "Pod":
        return doc["spec"]
    if kind in TEMPLATE_KINDS:
        return doc["spec"]["template"]["spec"]
    if kind == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    if kind in PODLESS_KINDS:
        return None
    raise ValueError(f"pod_spec: unknown kind {kind!r}; add it to TEMPLATE_KINDS or PODLESS_KINDS")


def all_containers(podspec) -> list[dict]:
    """Init containers followed by regular containers."""
    return [*podspec.get("initContainers", []), *podspec.get("containers", [])]
