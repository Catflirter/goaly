"""Stretch S4: k8s/scale-demo/cpu-hog.yaml, a Deployment that only reserves CPU so Pods go Pending for the CPU
node-layer demo (design spec section 10.2: another Deployment requests enough CPU that its Pods cannot be placed)."""
from k8s_helpers import K8S, all_containers, load_docs, one, pod_spec

CPU_HOG = K8S / "scale-demo" / "cpu-hog.yaml"
BUSYBOX = "docker.io/library/busybox:1.37@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e"


def test_cpu_hog_holds_only_the_deployment():
    docs = load_docs(CPU_HOG)
    assert [(d["kind"], d["metadata"]["name"]) for d in docs] == [("Deployment", "cpu-hog")]


def test_cpu_hog_contract():
    dep = one(load_docs(CPU_HOG), "Deployment", "cpu-hog")
    assert dep["metadata"]["namespace"] == "scale-demo"
    assert dep["spec"]["replicas"] == 0
    labels = dep["spec"]["template"]["metadata"]["labels"]
    assert dep["spec"]["selector"] == {"matchLabels": labels}
    (c,) = all_containers(pod_spec(dep))
    assert c["image"] == BUSYBOX
    assert c["command"] == ["sleep", "infinity"]
    want = {"cpu": "600m", "memory": "32Mi"}
    assert c["resources"] == {"requests": want, "limits": want}


def test_cpu_hog_meets_restricted():
    spec = pod_spec(one(load_docs(CPU_HOG), "Deployment", "cpu-hog"))
    psc = spec["securityContext"]
    assert psc["runAsNonRoot"] is True and psc["runAsUser"] > 0 and psc["runAsGroup"] > 0
    assert psc["seccompProfile"] == {"type": "RuntimeDefault"}
    assert spec["automountServiceAccountToken"] is False
    for key in ("hostNetwork", "hostPID", "hostIPC", "volumes"):
        assert key not in spec
    (c,) = all_containers(spec)
    assert c["securityContext"] == {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}
