"""Static guardrails over every Terraform file under infra/ (spec section 3.2, global constraints).

Files are discovered by glob, so stacks added later (workers, gpu-access) are checked too.
"""
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"

def tf_files():
    # .terraform/ holds downloaded providers and modules, not our code.
    return sorted(p for p in INFRA.rglob("*.tf") if ".terraform" not in p.relative_to(INFRA).parts)

def stacks():
    """A stack is a directory directly under infra/ that holds .tf files."""
    return sorted({p.relative_to(INFRA).parts[0] for p in tf_files() if len(p.relative_to(INFRA).parts) == 2})

def blocks(text, header_re):
    """Return each `<header> { ... }` block matching header_re, cut by brace matching.

    Interpolations ("${...}") are balanced, so they need no special casing; a literal
    unbalanced brace inside a string would confuse this, and our HCL has none.
    """
    out = []
    for m in re.finditer(header_re + r"\s*\{", text):
        depth, i = 0, m.end() - 1
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                out.append(text[m.start():j + 1])
                break
        else:
            raise AssertionError(f"unbalanced braces after {m.group(0)!r}")
    return out

def resource_blocks(rtype):
    return [(p, b) for p in tf_files() for b in blocks(p.read_text(), rf'resource\s+"{rtype}"\s+"[\w-]+"')]

def squash(s):
    return re.sub(r"\s+", "", s)

def test_infra_has_terraform_files():
    # Guards every glob-driven test below against passing vacuously.
    assert tf_files(), "no *.tf under infra/"
    assert {"bootstrap", "base"} <= set(stacks())

def test_blocks_helper_cuts_nested_blocks():
    text = 'resource "a" "x" {\n  b { c = "${d}" }\n}\nresource "a" "y" {}\n'
    assert blocks(text, r'resource\s+"a"\s+"\w+"') == [
        'resource "a" "x" {\n  b { c = "${d}" }\n}', 'resource "a" "y" {}']

def test_no_open_internet_source_anywhere():
    offenders = [str(p.relative_to(ROOT)) for p in tf_files() if "0.0.0.0/0" in p.read_text()]
    assert offenders == []

def test_no_iam_resources():
    # Service accounts get no IAM roles (spec section 4.5); no stack manages IAM at all.
    offenders = [str(p.relative_to(ROOT)) for p in tf_files()
                 if re.search(r'resource\s+"google_\w*_iam_', p.read_text())]
    assert offenders == []

def test_firewall_rules_log_and_target_service_accounts():
    rules = resource_blocks("google_compute_firewall")
    assert rules, "no google_compute_firewall resources found"
    for path, block in rules:
        where = f"{path.relative_to(ROOT)}: {block.splitlines()[0]}"
        assert "log_config" in block, where
        assert "target_service_accounts" in block, where
        assert "target_tags" not in block, where

def test_instances_ignore_boot_image_changes():
    instances = resource_blocks("google_compute_instance")
    assert instances, "no google_compute_instance resources found"
    for path, block in instances:
        assert "ignore_changes=[boot_disk[0].initialize_params[0].image]" in squash(block), (
            f"{path.relative_to(ROOT)}: {block.splitlines()[0]}")

@pytest.mark.parametrize("stack", stacks())
def test_stack_pins_terraform_and_provider_versions(stack):
    versions = INFRA / stack / "versions.tf"
    assert versions.is_file(), f"infra/{stack} has no versions.tf"
    text = versions.read_text()
    assert re.search(r'required_version\s*=\s*">= 1\.9\.0"', text)
    code = "".join(p.read_text() for p in tf_files() if p.parent == INFRA / stack)
    if re.search(r'(resource|data|provider)\s+"google', code):
        google = blocks(text, r"\bgoogle\s*=")
        assert len(google) == 1, f"infra/{stack}/versions.tf must declare the google provider once"
        assert re.search(r'version\s*=\s*"= 8\.7\.0"', google[0])
