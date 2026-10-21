"""Offline policy tests for the workload manifests (spec §7): nettest, hello, limits-demo, np-denied, gpu-test.

The verify/demo checks (Tasks 17/18) rely on the names, labels and ports asserted here.
"""
import re

import pytest

from k8s_helpers import K8S, all_containers, load_docs, one, pod_spec

FILES = {
    "nettest": K8S / "nettest" / "nettest.yaml",
    "hello": K8S / "hello" / "hello.yaml",
    "limits-demo": K8S / "limits-demo" / "limits-demo.yaml",
    "np-denied": K8S / "verify" / "np-denied.yaml",
    "gpu-namespace": K8S / "gpu-test" / "namespace.yaml",
    "gpu-smoke": K8S / "gpu-test" / "gpu-smoke.yaml",
    "vectoradd": K8S / "gpu-test" / "vectoradd.yaml",
    "gpu-hold": K8S / "gpu-test" / "gpu-hold.yaml",
    "gpu-overflow": K8S / "gpu-test" / "gpu-overflow.yaml",
}
WORKLOAD_FILES = [n for n in FILES if n != "gpu-namespace"]
GPU_FILES = ["gpu-smoke", "vectoradd", "gpu-hold", "gpu-overflow"]
WORKLOAD_NAMESPACES = {"nettest", "hello", "limits-demo", "np-denied", "gpu-test"}

# Registry digests of each tag's multi-arch index (docker buildx imagetools inspect), spec §1.2.
NETSHOOT = "docker.io/nicolaka/netshoot:v0.16@sha256:b09d9b21381f47a79b3cbcb30da25266dc17186ea00ae65e99fdc51396f48e70"
WHOAMI = "docker.io/traefik/whoami:v1.12.0@sha256:c4717a8d1f0134a7444e24f881160e033991f23027c6c5a9a3f8fd22e70d1d44"
HTTP_ECHO = "docker.io/hashicorp/http-echo:1.0.0@sha256:fcb75f691c8b0414d670ae570240cbf95502cc18a9ba57e982ecac589760a186"
CUDA_BASE = ("docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04"
             "@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2")
VECTORADD = ("nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0"
             "@sha256:7c7540bdf1f942d4fb6db97069fd6c289471b54ac29e3c7fcdf914cf77af7d41")
BUSYBOX = "docker.io/library/busybox:1.37@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e"


def docs(name) -> list[dict]:
    return load_docs(FILES[name])


def all_k8s_docs() -> list[dict]:
    """Every document under k8s/ except Helm values files (they are not Kubernetes objects)."""
    return [d for p in sorted(K8S.rglob("*.y*ml")) if not p.name.endswith("values.yaml") for d in load_docs(p)]


def pod_templates(name) -> list[tuple[dict, dict]]:
    """(document, PodSpec) for every Pod or Pod template in one workload file."""
    return [(d, pod_spec(d)) for d in docs(name) if pod_spec(d) is not None]


def test_all_images_pinned_by_digest():
    images = [c["image"] for doc in all_k8s_docs() for c in all_containers(pod_spec(doc) or {})]
    assert len(images) >= 10  # metrics-server + the 9 workload containers: the walk really reaches them
    for image in images:
        assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", image), image


def test_pod_spec_is_none_for_objects_without_pods():
    assert pod_spec({"kind": "Namespace", "metadata": {"name": "x"}}) is None
    assert pod_spec({"kind": "Service", "spec": {"ports": []}}) is None


@pytest.mark.parametrize("name", WORKLOAD_FILES)
def test_restricted_security_context_everywhere(name):
    templates = pod_templates(name)
    assert templates, f"{name} has no Pod templates"
    for doc, spec in templates:
        where = f"{name}: {doc['kind']}/{doc['metadata'].get('name') or doc['metadata'].get('generateName')}"
        psc = spec["securityContext"]
        assert psc["runAsNonRoot"] is True, where
        assert isinstance(psc["runAsUser"], int) and psc["runAsUser"] > 0, where
        assert isinstance(psc["runAsGroup"], int) and psc["runAsGroup"] > 0, where
        assert psc["seccompProfile"] == {"type": "RuntimeDefault"}, where
        for key in ("hostNetwork", "hostPID", "hostIPC"):
            assert not spec.get(key), where
        assert all(set(v) <= {"name", "emptyDir"} for v in spec.get("volumes", [])), where
        for c in all_containers(spec):
            csc = c["securityContext"]
            assert csc["allowPrivilegeEscalation"] is False, where
            assert csc["capabilities"] == {"drop": ["ALL"]}, where
            assert not csc.get("privileged"), where
            assert "runAsUser" not in csc and "runAsNonRoot" not in csc, where  # the Pod-level values apply
            assert not any("hostPort" in p for p in c.get("ports", [])), where


def test_workload_namespaces_enforce_restricted():
    namespaces = [d for name in FILES for d in docs(name) if d["kind"] == "Namespace"]
    assert sorted(ns["metadata"]["name"] for ns in namespaces) == sorted(WORKLOAD_NAMESPACES)
    for ns in namespaces:
        assert ns["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"
    for name in FILES:
        for d in docs(name):
            if d["kind"] != "Namespace":
                assert d["metadata"]["namespace"] in WORKLOAD_NAMESPACES, (name, d["kind"])


def test_nettest_echo_contract():
    ds = one(docs("nettest"), "DaemonSet", "echo"); spec = pod_spec(ds); c = spec["containers"][0]
    assert ds["metadata"]["namespace"] == "nettest"
    assert ds["spec"]["selector"]["matchLabels"] == {"app": "echo"} == ds["spec"]["template"]["metadata"]["labels"]
    assert c["image"] == WHOAMI
    assert {"name": "WHOAMI_PORT_NUMBER", "value": "8080"} in c["env"]
    assert spec["securityContext"]["runAsUser"] == 65532
    assert c["readinessProbe"]["httpGet"] == {"path": "/health", "port": 8080}
    assert {"containerPort": 8080, "name": "http", "protocol": "TCP"} in c["ports"]
    assert c["securityContext"]["readOnlyRootFilesystem"] is True


def test_nettest_probe_contract():
    ds = one(docs("nettest"), "DaemonSet", "probe"); spec = pod_spec(ds); c = spec["containers"][0]
    assert ds["metadata"]["namespace"] == "nettest"
    assert ds["spec"]["selector"]["matchLabels"] == {"app": "probe"} == ds["spec"]["template"]["metadata"]["labels"]
    assert c["image"] == NETSHOOT and c["command"] == ["sleep", "infinity"]
    assert c["resources"]["limits"] == {"cpu": "500m", "memory": "128Mi"}
    assert c["securityContext"]["readOnlyRootFilesystem"] is True
    assert c["volumeMounts"] == [{"name": "tmp", "mountPath": "/tmp"}]
    assert spec["volumes"] == [{"name": "tmp", "emptyDir": {}}]


def test_nettest_echo_service():
    svc = one(docs("nettest"), "Service", "echo")
    assert svc["metadata"]["namespace"] == "nettest" and svc["spec"]["selector"] == {"app": "echo"}
    assert svc["spec"]["ports"] == [{"name": "http", "port": 8080, "targetPort": 8080, "protocol": "TCP"}]


def test_nettest_tolerates_everything():
    for name in ("echo", "probe"):
        assert pod_spec(one(docs("nettest"), "DaemonSet", name))["tolerations"] == [{"operator": "Exists"}]


def test_hello_contract():
    dep = one(docs("hello"), "Deployment", "hello"); c = pod_spec(dep)["containers"][0]
    assert dep["spec"]["replicas"] == 2 and c["args"] == ["-text=hello world", "-listen=:8080"]
    assert c["image"] == HTTP_ECHO
    assert c["resources"] == {"requests": {"cpu": "100m", "memory": "64Mi"}, "limits": {"cpu": "100m", "memory": "64Mi"}}
    assert c["securityContext"]["readOnlyRootFilesystem"] is True and pod_spec(dep)["securityContext"]["runAsUser"] == 65532
    assert c["readinessProbe"]["httpGet"] == {"path": "/", "port": 8080} == c["livenessProbe"]["httpGet"]
    assert pod_spec(dep)["topologySpreadConstraints"][0]["whenUnsatisfiable"] == "ScheduleAnyway"
    assert dep["spec"]["selector"]["matchLabels"] == {"app": "hello"} == dep["spec"]["template"]["metadata"]["labels"]


def test_hello_service_80_to_8080():
    svc = one(docs("hello"), "Service", "hello")
    assert svc["metadata"]["namespace"] == "hello" and svc["spec"]["type"] == "ClusterIP"
    assert svc["spec"]["selector"] == {"app": "hello"}
    assert svc["spec"]["ports"] == [{"name": "http", "port": 80, "targetPort": 8080, "protocol": "TCP"}]


def test_hello_network_policies():
    policies = [d for d in docs("hello") if d["kind"] == "NetworkPolicy"]
    assert len(policies) == 2
    deny = one(policies, "NetworkPolicy", "default-deny-ingress")["spec"]
    assert deny == {"podSelector": {}, "policyTypes": ["Ingress"]}
    allow = one(policies, "NetworkPolicy", "allow-nettest-8080")["spec"]
    assert allow["podSelector"] == {"matchLabels": {"app": "hello"}} and allow["policyTypes"] == ["Ingress"]
    assert allow["ingress"] == [{
        "from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "nettest"}}}],
        "ports": [{"protocol": "TCP", "port": 8080}],
    }]


def test_limits_demo_contract():
    d = docs("limits-demo")
    oom, burn = one(d, "Pod", "oom"), one(d, "Pod", "cpu-burn")
    assert oom["metadata"]["namespace"] == burn["metadata"]["namespace"] == "limits-demo"
    oc, bc = oom["spec"]["containers"][0], burn["spec"]["containers"][0]
    assert oc["image"] == BUSYBOX == bc["image"]
    # restartPolicy Never: the Pod stays OOMKilled (exit 137) instead of crash-looping (spec §7).
    assert oom["spec"]["restartPolicy"] == "Never" and oc["command"] == ["tail", "/dev/zero"]
    assert oc["resources"]["limits"]["memory"] == "64Mi"
    assert bc["resources"]["limits"]["cpu"] == "100m"
    assert bc["command"][:2] == ["sh", "-c"] and "while" in bc["command"][2]


def test_np_denied_client_contract():
    client = one(docs("np-denied"), "Pod", "client")
    assert client["metadata"]["namespace"] == "np-denied"
    c = client["spec"]["containers"][0]
    assert c["image"] == NETSHOOT and c["command"] == ["sleep", "infinity"]  # netshoot ships curl for check C8


@pytest.mark.parametrize("name", GPU_FILES)
def test_gpu_pods_contract(name):
    d = docs(name)
    assert len(d) == 1 and d[0]["kind"] == "Pod"
    pod = d[0]; spec = pod["spec"]; c = spec["containers"][0]
    assert pod["metadata"]["namespace"] == "gpu-test"
    assert spec["runtimeClassName"] == "nvidia-gpu"
    assert c["resources"] == {"limits": {"nvidia.com/gpu": 1}}
    assert spec["securityContext"]["runAsUser"] == 1000
    assert spec["restartPolicy"] == "Never"
    for ctr in all_containers(spec):  # the device plugin and runtime choose the GPU, never the Pod (spec §7)
        assert all(e["name"] != "NVIDIA_VISIBLE_DEVICES" for e in ctr.get("env", [])), name


def test_gpu_pod_identities():
    assert one(docs("gpu-smoke"), "Pod", "gpu-smoke")["spec"]["containers"][0]["command"] == ["nvidia-smi"]
    assert one(docs("gpu-smoke"), "Pod", "gpu-smoke")["spec"]["containers"][0]["image"] == CUDA_BASE
    assert one(docs("vectoradd"), "Pod", "vectoradd")["spec"]["containers"][0]["image"] == VECTORADD
    overflow = one(docs("gpu-overflow"), "Pod", "gpu-overflow")["spec"]["containers"][0]
    assert overflow["image"] == CUDA_BASE and overflow["command"] == ["sleep", "60"]


def test_gpu_hold_is_repeatable_with_kubectl_create():
    meta = docs("gpu-hold")[0]["metadata"]
    # generateName (no fixed name) lets `kubectl create -f` make one holder per allocatable GPU (check D6).
    assert meta["generateName"] == "gpu-hold-" and "name" not in meta
    assert meta["labels"] == {"app": "gpu-hold"}
    c = docs("gpu-hold")[0]["spec"]["containers"][0]
    assert c["image"] == CUDA_BASE and c["command"] == ["sleep", "600"]


def test_gpu_namespace_file_holds_only_the_namespace():
    assert [(d["kind"], d["metadata"]["name"]) for d in docs("gpu-namespace")] == [("Namespace", "gpu-test")]
