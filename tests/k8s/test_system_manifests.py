"""Offline policy tests for the system layer in k8s/system/ (metrics-server, RuntimeClass, device plugin)."""
import re

import pytest

from k8s_helpers import K8S, all_containers, load_docs, one, pod_spec

SYS = K8S / "system"


def test_metrics_server_pinned_to_cp():
    dep = one(load_docs(SYS / "metrics-server.yaml"), "Deployment", "metrics-server")
    spec = pod_spec(dep); c = spec["containers"][0]
    assert spec["nodeSelector"] == {"node-role.kubernetes.io/control-plane": "true"}
    assert {"key": "CriticalAddonsOnly", "operator": "Exists"} in spec["tolerations"]
    assert "--kubelet-preferred-address-types=ExternalIP,InternalIP,Hostname" in c["args"]
    assert re.fullmatch(r"docker\.io/rancher/mirrored-metrics-server:v[\d.]+@sha256:[0-9a-f]{64}", c["image"])


def test_metrics_server_keeps_k3s_tag_and_upstream_toleration():
    # Global constraint: the same tag as k3s v1.36.5+k3s1's bundled manifest; its original tolerations stay.
    spec = pod_spec(one(load_docs(SYS / "metrics-server.yaml"), "Deployment", "metrics-server"))
    assert spec["containers"][0]["image"].startswith("docker.io/rancher/mirrored-metrics-server:v0.9.0@sha256:")
    assert {"key": "node-role.kubernetes.io/control-plane", "operator": "Exists", "effect": "NoSchedule"} \
        in spec["tolerations"]


def test_metrics_server_bundle_is_complete():
    # The k3s bundle is seven files; all of their objects must be in the one concatenated manifest.
    got = {(d["kind"], d["metadata"]["name"]) for d in load_docs(SYS / "metrics-server.yaml")}
    assert got == {
        ("ClusterRole", "system:aggregated-metrics-reader"),
        ("ClusterRoleBinding", "metrics-server:system:auth-delegator"),
        ("RoleBinding", "metrics-server-auth-reader"),
        ("APIService", "v1beta1.metrics.k8s.io"),
        ("ServiceAccount", "metrics-server"),
        ("Deployment", "metrics-server"),
        ("Service", "metrics-server"),
        ("ClusterRole", "system:metrics-server"),
        ("ClusterRoleBinding", "system:metrics-server"),
    }


def test_no_k3s_template_placeholders_left():
    assert "%{" not in (SYS / "metrics-server.yaml").read_text()


def test_runtimeclass_matches_spec():
    assert load_docs(SYS / "nvidia-gpu-runtimeclass.yaml") == [{"apiVersion": "node.k8s.io/v1", "kind": "RuntimeClass",
        "metadata": {"name": "nvidia-gpu"}, "handler": "nvidia", "scheduling": {
        "nodeSelector": {"node-restriction.kubernetes.io/accelerator": "nvidia"},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]}}]


def test_nvdp_namespace_is_privileged():
    docs = load_docs(SYS / "nvidia-device-plugin-namespace.yaml")
    assert len(docs) == 1
    ns = one(docs, "Namespace", "nvidia-device-plugin")
    # warn and audit as well: otherwise they fall back to the cluster-wide restricted default (node/k3s/psa.yaml),
    # and every helm upgrade of the hostPath DaemonSets prints PodSecurity warnings and writes audit annotations.
    assert ns["metadata"]["labels"] == {f"pod-security.kubernetes.io/{mode}": "privileged"
                                        for mode in ("enforce", "warn", "audit")}


# Index digest of nvcr.io/nvidia/k8s-device-plugin:v0.20.1 (the chart 0.20.1 appVersion).
NVDP_DIGEST = "sha256:27c1b2553a690ca3d29889dcd61df97525d46f58272c0be8dc63c2b18f120075"


def test_nvdp_values():
    # Tolerations are explicit so the DaemonSet does not depend on the chart's defaults (spec §12 #10).
    # The image is digest-pinned (spec §1.2). The chart renders repository:tag, so the digest goes in image.tag.
    assert load_docs(SYS / "nvdp-values.yaml") == [{
        "image": {"tag": f"v0.20.1@{NVDP_DIGEST}"},
        "runtimeClassName": "nvidia-gpu",
        "tolerations": [
            {"key": "CriticalAddonsOnly", "operator": "Exists"},
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        ],
    }]


def test_nvdp_image_tag_is_digest_pinned():
    # Spec §1.2: every image is pinned by digest; the tag stays the spec's v0.20.1.
    tag = load_docs(SYS / "nvdp-values.yaml")[0]["image"]["tag"]
    assert re.fullmatch(r"v0\.20\.1@sha256:[0-9a-f]{64}", tag)


_PODSPEC = {"containers": [{"name": "c"}], "initContainers": [{"name": "i"}]}


@pytest.mark.parametrize("doc", [
    {"kind": "Pod", "spec": _PODSPEC},
    {"kind": "Deployment", "spec": {"template": {"spec": _PODSPEC}}},
    {"kind": "DaemonSet", "spec": {"template": {"spec": _PODSPEC}}},
    {"kind": "Job", "spec": {"template": {"spec": _PODSPEC}}},
    {"kind": "CronJob", "spec": {"jobTemplate": {"spec": {"template": {"spec": _PODSPEC}}}}},
])
def test_helpers_reach_the_pod_spec(doc):
    assert [c["name"] for c in all_containers(pod_spec(doc))] == ["i", "c"]
