"""Offline policy tests for the monitoring stack in k8s/monitoring/ (spec §10.0, stretch S0).

Prometheus, kube-state-metrics and Grafana run in the restricted namespace `monitoring`; node-exporter and
the DCGM exporter need hostPath mounts and run in the separate privileged namespace `monitoring-host`.
Everything is scraped over the Pod network (WireGuard) and reached by the operator only through
`kubectl port-forward` (scripts/monitoring.sh), so no Service is exposed and no firewall rule changes.
"""
import json
import re

import pytest
import yaml

from k8s_helpers import K8S, all_containers, load_docs, one, pod_spec

MON = K8S / "monitoring"
MANIFESTS = ["namespaces.yaml", "prometheus.yaml", "kube-state-metrics.yaml", "node-exporter.yaml",
             "dcgm-exporter.yaml", "grafana.yaml", "network-policies.yaml"]
DASHBOARD = MON / "grafana-dashboard.json"
NS, HOST_NS = "monitoring", "monitoring-host"
CLUSTER_SCOPED = {"Namespace", "ClusterRole", "ClusterRoleBinding"}
# The CPU worker that is never removed (global constraints): the monitoring Deployments stay on it, so they
# never keep a scaled-up worker busy (stretch Review Focus #3/#5: an extra worker must be able to go idle).
ANCHOR_NODE = "k3s-cpu-1"
GPU_LABEL = "node-restriction.kubernetes.io/accelerator"

# Registry digests of each tag's multi-arch index (spec §1.2: every image pinned by digest).
IMAGES = {
    "prometheus": "quay.io/prometheus/prometheus:v3.15.1"
                  "@sha256:fa3302279710be2f21a8301ec17a8b4f2301fd829bd113839b3b6b02b6f191a2",
    "node-exporter": "quay.io/prometheus/node-exporter:v1.12.1"
                     "@sha256:1b4e4438faca4dd7e001dd445d161a4a2091b0fededa84093b3a8dfeae1f1be0",
    "kube-state-metrics": "registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.20.0"
                          "@sha256:42cfe3723a5f058171c627537fb57a3ea0f26e4380fa18555a95cb1a1b4cfc5b",
    "dcgm-exporter": "nvcr.io/nvidia/k8s/dcgm-exporter:4.5.2-4.8.1-ubuntu22.04"
                     "@sha256:17e9d49093f2d86d90260eec3c07b18b3f78fc5eadef3c8ab26a24fa5a9c1b2c",
    "grafana": "docker.io/grafana/grafana:13.2.4"
               "@sha256:dbbc38ea21103cefb50396340cd88afec779c7e070b0a7bf5a75b4ce46651d3f",
}


def docs(name) -> list[dict]:
    return load_docs(MON / name)


def all_docs() -> list[dict]:
    return [d for name in MANIFESTS for d in docs(name)]


def workloads() -> list[dict]:
    return [d for d in all_docs() if pod_spec(d) is not None]


def workload(kind, name) -> dict:
    return one(all_docs(), kind, name)


def configmap_data(name, key) -> str:
    return one(all_docs(), "ConfigMap", name)["data"][key]


def prometheus_config() -> dict:
    return yaml.safe_load(configmap_data("prometheus-config", "prometheus.yml"))


def env_of(container) -> dict:
    return {e["name"]: e for e in container.get("env", [])}


# ---- files, images, namespaces ------------------------------------------------------------------

def test_directory_holds_exactly_the_listed_files():
    # scripts/monitoring.sh applies MANIFESTS by name; a new file must be added there and here.
    assert sorted(p.name for p in MON.iterdir()) == sorted([*MANIFESTS, DASHBOARD.name])


def test_every_image_is_the_pinned_digest():
    images = {c["name"]: c["image"] for d in workloads() for c in all_containers(pod_spec(d))}
    assert images == IMAGES
    for image in images.values():
        assert re.fullmatch(r"[a-z0-9.\-/]+:[\w.\-]+@sha256:[0-9a-f]{64}", image), image


def test_namespaces_and_their_pod_security_levels():
    namespaces = {d["metadata"]["name"]: d["metadata"]["labels"] for d in docs("namespaces.yaml")}
    assert namespaces == {
        NS: {f"pod-security.kubernetes.io/{m}": "restricted" for m in ("enforce", "warn", "audit")},
        HOST_NS: {f"pod-security.kubernetes.io/{m}": "privileged" for m in ("enforce", "warn", "audit")},
    }
    # The exception is justified next to the label, as for nvidia-device-plugin (spec §4.3).
    text = (MON / "namespaces.yaml").read_text()
    assert "hostPath" in text and "privileged" in text


def test_every_object_lives_in_a_monitoring_namespace():
    for d in all_docs():
        if d["kind"] in CLUSTER_SCOPED:
            assert "namespace" not in d["metadata"], d["metadata"]["name"]
        else:
            assert d["metadata"]["namespace"] in {NS, HOST_NS}, (d["kind"], d["metadata"]["name"])


def test_workloads_and_their_namespaces():
    got = {(d["kind"], d["metadata"]["name"]): d["metadata"]["namespace"] for d in workloads()}
    assert got == {
        ("Deployment", "prometheus"): NS,
        ("Deployment", "kube-state-metrics"): NS,
        ("Deployment", "grafana"): NS,
        ("DaemonSet", "node-exporter"): HOST_NS,
        ("DaemonSet", "dcgm-exporter"): HOST_NS,
    }


# ---- Pod security, resources, exposure ---------------------------------------------------------

def test_every_container_has_explicit_requests_and_limits():
    for d in workloads():
        for c in all_containers(pod_spec(d)):
            res = c["resources"]
            for part in ("requests", "limits"):
                assert set(res[part]) >= {"cpu", "memory"}, (c["name"], part)


def _quantity(value) -> float:
    """CPU in cores or memory in MiB, from the few Kubernetes quantity forms these manifests use."""
    value = str(value)
    for suffix, factor in (("m", 1e-3), ("Mi", 1), ("Gi", 1024)):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * factor
    return float(value)


def test_requests_on_the_permanent_worker_fit_next_to_the_workloads():
    # k3s-cpu-1 becomes an e2-standard-2 (2 vCPU, 8 GiB) for S0 and also runs hello, nettest and the
    # HPA demo: the monitoring stack's requests there (three Deployments + one node-exporter) stay small.
    on_worker = [workload("Deployment", n) for n in ("prometheus", "kube-state-metrics", "grafana")]
    on_worker.append(workload("DaemonSet", "node-exporter"))
    containers = [c for d in on_worker for c in all_containers(pod_spec(d))]
    cpu = sum(_quantity(c["resources"]["requests"]["cpu"]) for c in containers)
    mem = sum(_quantity(c["resources"]["requests"]["memory"]) for c in containers)
    assert cpu <= 0.25 and mem <= 1024, (cpu, mem)
    for c in containers:  # never a request above its limit
        for r in ("cpu", "memory"):
            assert _quantity(c["resources"]["requests"][r]) <= _quantity(c["resources"]["limits"][r]), c["name"]


def test_no_host_namespaces_ports_or_privilege_anywhere():
    for d in workloads():
        spec = pod_spec(d)
        # No hostNetwork: a node IP:port would need a new firewall rule; Pod IPs travel over WireGuard.
        for key in ("hostNetwork", "hostPID", "hostIPC"):
            assert not spec.get(key), (d["metadata"]["name"], key)
        for c in all_containers(spec):
            csc = c["securityContext"]
            assert csc["allowPrivilegeEscalation"] is False, c["name"]
            assert csc["capabilities"] == {"drop": ["ALL"]}, c["name"]
            assert not csc.get("privileged"), c["name"]
            assert not any("hostPort" in p for p in c.get("ports", [])), c["name"]


@pytest.mark.parametrize("name", ["prometheus", "kube-state-metrics", "grafana"])
def test_restricted_namespace_workloads_meet_psa_restricted(name):
    spec = pod_spec(workload("Deployment", name))
    psc = spec["securityContext"]
    assert psc["runAsNonRoot"] is True
    assert isinstance(psc["runAsUser"], int) and psc["runAsUser"] > 0
    assert psc["seccompProfile"] == {"type": "RuntimeDefault"}
    allowed = {"name", "emptyDir", "configMap", "secret", "projected"}
    assert all(set(v) <= allowed for v in spec.get("volumes", [])), name
    for c in all_containers(spec):
        assert c["securityContext"]["readOnlyRootFilesystem"] is True


def test_host_paths_only_in_the_privileged_namespace_and_read_only():
    for d in workloads():
        spec = pod_spec(d)
        host_volumes = {v["name"] for v in spec.get("volumes", []) if "hostPath" in v}
        if d["metadata"]["namespace"] == NS:
            assert not host_volumes, d["metadata"]["name"]
        for c in all_containers(spec):
            for m in c.get("volumeMounts", []):
                if m["name"] in host_volumes:
                    assert m.get("readOnly") is True, (c["name"], m["name"])


def test_services_are_cluster_ip_only_and_nothing_is_exposed():
    kinds = {d["kind"] for d in all_docs()}
    assert not kinds & {"Ingress", "IngressRoute", "Gateway", "HTTPRoute", "Secret"}
    services = [d for d in all_docs() if d["kind"] == "Service"]
    assert {s["metadata"]["name"] for s in services} == {"prometheus", "grafana"}
    for s in services:
        assert s["spec"].get("type", "ClusterIP") == "ClusterIP"
        assert not s["spec"].get("externalIPs")
        assert all("nodePort" not in p for p in s["spec"]["ports"])


def test_only_prometheus_and_kube_state_metrics_get_an_api_token():
    for d in workloads():
        spec = pod_spec(d)
        name = d["metadata"]["name"]
        assert spec["serviceAccountName"] == name
        assert spec["automountServiceAccountToken"] is (name in {"prometheus", "kube-state-metrics"}), name


# ---- scheduling ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["prometheus", "kube-state-metrics", "grafana"])
def test_deployments_run_on_the_permanent_cpu_worker(name):
    dep = workload("Deployment", name)
    spec = pod_spec(dep)
    assert dep["spec"]["replicas"] == 1
    assert spec["nodeSelector"] == {"kubernetes.io/hostname": ANCHOR_NODE}
    # No tolerations: never on k3s-cp (CriticalAddonsOnly NoExecute) or a GPU node (nvidia.com/gpu taint).
    assert not spec.get("tolerations")
    assert "runtimeClassName" not in spec


def test_node_exporter_runs_on_every_node():
    spec = pod_spec(workload("DaemonSet", "node-exporter"))
    assert spec["tolerations"] == [{"operator": "Exists"}]  # cp, GPU nodes, cordoned/not-ready nodes too
    assert "nodeSelector" not in spec and "affinity" not in spec
    assert "runtimeClassName" not in spec
    psc = spec["securityContext"]
    assert psc["runAsNonRoot"] is True and psc["runAsUser"] == 65534
    assert psc["seccompProfile"] == {"type": "RuntimeDefault"}
    c = spec["containers"][0]
    assert c["securityContext"]["readOnlyRootFilesystem"] is True
    args = c["args"]
    for flag in ("--path.procfs=/host/proc", "--path.sysfs=/host/sys", "--path.rootfs=/host/root"):
        assert flag in args
    # Without hostNetwork these collectors would report the Pod's own network namespace: switched off.
    for collector in ("netdev", "netstat", "sockstat", "netclass", "arp", "conntrack"):
        assert f"--no-collector.{collector}" in args
    assert {v["hostPath"]["path"] for v in spec["volumes"] if "hostPath" in v} == {"/proc", "/sys", "/"}


def test_dcgm_exporter_only_on_admin_labelled_gpu_nodes():
    spec = pod_spec(workload("DaemonSet", "dcgm-exporter"))
    assert spec["nodeSelector"] == {GPU_LABEL: "nvidia"}
    assert spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert spec["runtimeClassName"] == "nvidia-gpu"
    assert "affinity" not in spec


def test_dcgm_exporter_never_takes_a_gpu():
    # It watches every GPU through NVIDIA_VISIBLE_DEVICES=all (spec §6.4's mechanism, used on purpose)
    # and requests no nvidia.com/gpu, so the device plugin never hands it a GPU a workload needs.
    spec = pod_spec(workload("DaemonSet", "dcgm-exporter"))
    for c in all_containers(spec):
        assert set(c["resources"]) == {"requests", "limits"}, c["name"]
        for part in ("requests", "limits"):
            assert set(c["resources"][part]) == {"cpu", "memory"}, (c["name"], part)
    c = spec["containers"][0]
    assert env_of(c)["NVIDIA_VISIBLE_DEVICES"]["value"] == "all"


def test_dcgm_exporter_has_no_extra_privileges():
    spec = pod_spec(workload("DaemonSet", "dcgm-exporter"))
    c = spec["containers"][0]
    # Upstream's chart adds SYS_ADMIN for the profiling (DCP) metrics; the dashboard needs none of them.
    assert c["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    hostpaths = {v["hostPath"]["path"] for v in spec["volumes"] if "hostPath" in v}
    assert hostpaths == {"/var/lib/kubelet/pod-resources"}
    assert c["args"] == ["-f", "/etc/dcgm-exporter/counters.csv"]


def test_dcgm_counters_cover_the_dashboard_and_skip_profiling_metrics():
    rows = [line.split(",")[0].strip() for line in configmap_data("dcgm-counters", "counters.csv").splitlines()
            if line.strip() and not line.startswith("#")]
    assert {"DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_GPU_TEMP"} <= set(rows)
    assert not [r for r in rows if r.startswith("DCGM_FI_PROF_")]


# ---- Prometheus ---------------------------------------------------------------------------------

def test_prometheus_flags():
    c = pod_spec(workload("Deployment", "prometheus"))["containers"][0]
    args = c["args"]
    assert "--config.file=/etc/prometheus/prometheus.yml" in args
    assert "--config.auto-reload" in args  # a changed ConfigMap is picked up without a reloader sidecar
    # Neither HTTP API that changes state: anyone who can reach the Pod could shut it down or delete data.
    assert not [a for a in args if a.startswith(("--web.enable-lifecycle", "--web.enable-admin-api"))]


def test_prometheus_keeps_24h_in_a_bounded_empty_dir():
    cfg = prometheus_config()
    assert cfg["storage"]["tsdb"]["retention"] == {"time": "24h", "size": "1GB"}
    spec = pod_spec(workload("Deployment", "prometheus"))
    data = one([{"kind": "v", "metadata": v} for v in spec["volumes"]], "v", "data")["metadata"]
    assert data["emptyDir"]["sizeLimit"] == "2Gi"
    assert workload("Deployment", "prometheus")["spec"]["strategy"] == {"type": "Recreate"}


def test_prometheus_scrapes_pods_over_the_pod_network_only():
    cfg = prometheus_config()
    jobs = {j["job_name"]: j for j in cfg["scrape_configs"]}
    assert set(jobs) == {"prometheus", "node-exporter", "kube-state-metrics", "dcgm-exporter"}
    want_ns = {"node-exporter": HOST_NS, "dcgm-exporter": HOST_NS, "kube-state-metrics": NS}
    for job, ns in want_ns.items():
        sd = jobs[job]["kubernetes_sd_configs"]
        assert sd == [{"role": "pod", "namespaces": {"names": [ns]}}], job
        assert jobs[job].get("scheme", "http") == "http"
    # No kubelet/cAdvisor scraping: worker -> node:10250 is not in the firewall model, and the API server
    # proxy path would need nodes/proxy, which lets its holder run commands through the kubelet.
    text = configmap_data("prometheus-config", "prometheus.yml")
    assert "10250" not in text and "role: node" not in text and "proxy" not in text


def test_prometheus_rbac_is_read_only_pods_in_the_two_namespaces():
    roles = [d for d in all_docs() if d["kind"] == "Role"]
    assert {(r["metadata"]["namespace"], r["metadata"]["name"]) for r in roles} == {
        (NS, "prometheus-discovery"), (HOST_NS, "prometheus-discovery")}
    for r in roles:
        assert r["rules"] == [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}]
    bindings = [d for d in all_docs() if d["kind"] == "RoleBinding"]
    assert len(bindings) == 2
    for b in bindings:
        assert b["roleRef"] == {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role",
                                "name": "prometheus-discovery"}
        assert b["subjects"] == [{"kind": "ServiceAccount", "name": "prometheus", "namespace": NS}]


def test_no_rbac_grants_secrets_proxy_or_writes():
    for d in all_docs():
        for rule in d.get("rules", []) if d["kind"] in {"Role", "ClusterRole"} else []:
            assert set(rule["verbs"]) <= {"get", "list", "watch"}, d["metadata"]["name"]
            for res in rule["resources"]:
                assert res not in {"secrets", "configmaps", "*"} and "proxy" not in res, d["metadata"]["name"]


def test_kube_state_metrics_resources_match_its_rbac():
    c = pod_spec(workload("Deployment", "kube-state-metrics"))["containers"][0]
    resources = next(a for a in c["args"] if a.startswith("--resources=")).split("=", 1)[1].split(",")
    assert {"deployments", "pods", "nodes", "daemonsets", "horizontalpodautoscalers"} <= set(resources)
    role = workload("ClusterRole", "kube-state-metrics")
    granted = {res for rule in role["rules"] for res in rule["resources"]}
    assert set(resources) == granted
    binding = workload("ClusterRoleBinding", "kube-state-metrics")
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "kube-state-metrics", "namespace": NS}]


# ---- Grafana ------------------------------------------------------------------------------------

def test_grafana_admin_credentials_come_from_the_secret():
    c = pod_spec(workload("Deployment", "grafana"))["containers"][0]
    env = env_of(c)
    for var, key in (("GF_SECURITY_ADMIN_USER", "admin-user"), ("GF_SECURITY_ADMIN_PASSWORD", "admin-password"),
                     ("GF_SECURITY_SECRET_KEY", "secret-key")):
        assert env[var] == {"name": var, "valueFrom": {"secretKeyRef": {"name": "grafana-admin", "key": key}}}
    # No literal credential anywhere in the environment.
    for name, e in env.items():
        if re.search(r"PASSWORD|SECRET|TOKEN|KEY", name):
            assert "value" not in e, name


def test_grafana_anonymous_access_and_phone_home_are_off():
    env = {k: e.get("value") for k, e in env_of(pod_spec(workload("Deployment", "grafana"))["containers"][0]).items()}
    assert env["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert env["GF_USERS_ALLOW_SIGN_UP"] == "false"
    assert env["GF_PLUGINS_PREINSTALL_DISABLED"] == "true"  # no plugin downloads at start
    assert env["GF_ANALYTICS_REPORTING_ENABLED"] == "false"
    assert env["GF_ANALYTICS_CHECK_FOR_UPDATES"] == "false"


def test_grafana_provisions_the_datasource_and_the_dashboard():
    ds = yaml.safe_load(configmap_data("grafana-datasources", "datasources.yaml"))["datasources"]
    assert ds == [{"name": "Prometheus", "uid": "prometheus", "type": "prometheus", "access": "proxy",
                   "url": "http://prometheus.monitoring.svc:9090", "isDefault": True, "editable": False,
                   "jsonData": {"timeInterval": "15s"}}]
    provider = yaml.safe_load(configmap_data("grafana-dashboard-provider", "dashboards.yaml"))["providers"][0]
    spec = pod_spec(workload("Deployment", "grafana"))
    mounts = {m["name"]: m["mountPath"] for m in spec["containers"][0]["volumeMounts"]}
    volumes = {v["name"]: v for v in spec["volumes"]}
    assert provider["options"]["path"] == mounts["dashboards"]
    # scripts/monitoring.sh builds this ConfigMap from grafana-dashboard.json.
    assert volumes["dashboards"]["configMap"]["name"] == "grafana-dashboards"
    assert mounts["datasources"] == "/etc/grafana/provisioning/datasources"
    assert mounts["dashboard-provider"] == "/etc/grafana/provisioning/dashboards"


def test_dashboard_shows_nodes_gpus_and_replicas():
    dash = json.loads(DASHBOARD.read_text())
    assert dash["uid"] == "hybrid-gpu"
    exprs = " ".join(t["expr"] for p in dash["panels"] for t in p.get("targets", []))
    for metric in ("node_cpu_seconds_total", "node_memory_MemAvailable_bytes", "DCGM_FI_DEV_GPU_UTIL",
                   "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_GPU_TEMP", "kube_deployment_status_replicas_available"):
        assert metric in exprs, metric
    for p in dash["panels"]:
        if p["type"] == "row":
            continue
        assert p["datasource"] == {"type": "prometheus", "uid": "prometheus"}, p["title"]
        assert p["fieldConfig"]["defaults"].get("unit"), p["title"]


# ---- NetworkPolicies ----------------------------------------------------------------------------

def test_default_deny_ingress_in_both_namespaces():
    nps = [d for d in docs("network-policies.yaml") if d["kind"] == "NetworkPolicy"]
    for ns in (NS, HOST_NS):
        deny = one([n for n in nps if n["metadata"]["namespace"] == ns], "NetworkPolicy", "default-deny-ingress")
        assert deny["spec"] == {"podSelector": {}, "policyTypes": ["Ingress"]}


def _from_prometheus():
    return {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": NS}},
            "podSelector": {"matchLabels": {"app.kubernetes.io/name": "prometheus"}}}


@pytest.mark.parametrize("ns,name,target,port,source", [
    (HOST_NS, "allow-prometheus-node-exporter", "node-exporter", 9100, "prometheus"),
    (HOST_NS, "allow-prometheus-dcgm-exporter", "dcgm-exporter", 9400, "prometheus"),
    (NS, "allow-prometheus-kube-state-metrics", "kube-state-metrics", 8080, "prometheus"),
    (NS, "allow-grafana-prometheus", "prometheus", 9090, "grafana"),
])
def test_allow_rules(ns, name, target, port, source):
    np = one(docs("network-policies.yaml"), "NetworkPolicy", name)
    assert np["metadata"]["namespace"] == ns
    assert np["spec"]["podSelector"] == {"matchLabels": {"app.kubernetes.io/name": target}}
    assert np["spec"]["policyTypes"] == ["Ingress"]
    peer = _from_prometheus() if source == "prometheus" else {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": NS}},
        "podSelector": {"matchLabels": {"app.kubernetes.io/name": "grafana"}}}
    assert np["spec"]["ingress"] == [{"from": [peer], "ports": [{"protocol": "TCP", "port": port}]}]


def test_scraped_ports_match_the_network_policies():
    ports = {d["metadata"]["name"]: {p["containerPort"] for c in pod_spec(d)["containers"] for p in c["ports"]}
             for d in workloads()}
    assert 9100 in ports["node-exporter"] and 9400 in ports["dcgm-exporter"]
    assert 8080 in ports["kube-state-metrics"] and 9090 in ports["prometheus"] and 3000 in ports["grafana"]
    for d in workloads():
        assert d["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/name"] == d["metadata"]["name"]
