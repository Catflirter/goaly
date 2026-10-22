"""Delivery documents (Task 23): README, the six ADRs (MADR), threat model, runbook and P2 roadmap."""
import ipaddress
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ADRS = sorted((ROOT / "docs/adr").glob("*.md"))
DOCS = [ROOT / "README.md", *ADRS, *(ROOT / "docs").glob("*.md")]


def test_six_adrs_follow_madr():
    assert [p.name for p in ADRS] == ["0001-k3s-not-kubeadm.md", "0002-network-k3s-wireguard-native.md",
        "0003-terraform-three-stacks.md", "0004-gpu-nodes-as-cattle-spec-reconciler.md",
        "0005-gpu-device-plugin-and-runtimeclass.md", "0006-ip-allowlist-vs-identity-network.md"]
    for p in ADRS:
        t = p.read_text()
        for h in ("## Context and Problem Statement", "## Considered Options", "## Decision Outcome", "### Consequences"):
            assert h in t, (p.name, h)
        assert re.search(r"(?im)^status:\s*accepted", t)


def test_readme_sections_and_diagram():
    t = (ROOT / "README.md").read_text()
    for h in ("## Architecture", "## Networking and security", "## GPU enablement", "## Reproduce",
              "## Verification", "## Live demo", "## Cost", "## Limitations and next steps"):
        assert h in t
    assert "```mermaid" in t and "verification/final/SUMMARY.md" in t


def test_relative_links_resolve():
    for p in DOCS:
        for target in re.findall(r"\]\((?!https?://|#)([^)#]+)", p.read_text()):
            assert (p.parent / target).resolve().exists(), (p.name, target)


def test_no_placeholders():
    for p in DOCS:
        assert not re.search(r"\b(TODO|TBD|FIXME)\b|lorem ipsum", p.read_text(), re.I), p.name


def test_threat_model_runbook_p2_sections():
    tm = (ROOT / "docs/threat-model.md").read_text()
    for h in ("## Assets", "## Trust boundaries", "## Threats and mitigations", "## Residual risks"): assert h in tm
    rb = (ROOT / "docs/runbook.md").read_text()
    for h in ("## Prerequisites", "## Bring-up", "## Scaling", "## GPU nodes", "## Break-glass", "## Teardown",
              "## Demo-day checklist", "## Measured timings"): assert h in rb
    p2 = (ROOT / "docs/p2-roadmap.md").read_text()
    for h in ("Networking and scale", "High availability", "Security", "GPU", "Operations"): assert h in p2


# --- beyond the brief ------------------------------------------------------------------------------

# Public addresses a document may name: Google's IAP TCP-forwarding range (the one non-/32 firewall
# source) and the RFC 5737 documentation ranges used in examples. Every other address must be private,
# loopback or link-local (the metadata server); a real node, operator or Lambda IP never belongs in a doc.
ALLOWED_PUBLIC = [ipaddress.ip_network(n) for n in
                  ("35.235.240.0/20", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?:/\d{1,2})?(?![\d.]*\d)")


def test_no_real_public_ip_addresses():
    for p in DOCS:
        for text in IPV4.findall(p.read_text()):
            try:
                ip = ipaddress.ip_address(text)
            except ValueError:
                continue  # not an address (e.g. a version number with four parts)
            ok = ip.is_private or ip.is_loopback or ip.is_link_local or any(ip in n for n in ALLOWED_PUBLIC)
            assert ok, (p.name, text)


def test_ip_check_rejects_a_public_address():
    # The guard above must be able to fail: a public address outside the allowed ranges is reported.
    found = IPV4.findall("node at 34.82.10.10, docs at 203.0.113.7/32, version 1.18.1, pod 10.42.0.0/16")
    assert found == ["34.82.10.10", "203.0.113.7", "10.42.0.0"]
    assert not ipaddress.ip_address("34.82.10.10").is_private


def test_readme_links_every_adr_and_doc():
    t = (ROOT / "README.md").read_text()
    for p in [*ADRS, ROOT / "docs/threat-model.md", ROOT / "docs/runbook.md", ROOT / "docs/p2-roadmap.md"]:
        assert f"]({p.relative_to(ROOT)})" in t, p.name


def test_mermaid_blocks_name_a_diagram_type():
    for p in DOCS:
        for block in re.findall(r"```mermaid\n(.*?)```", p.read_text(), re.S):
            first = block.strip().splitlines()[0]
            assert re.match(r"(flowchart|graph) (LR|RL|TB|TD|BT)$", first), (p.name, first)
