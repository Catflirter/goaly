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

_HEREDOC = re.compile(r"<<-?([A-Za-z_][\w-]*)[ \t]*\n")

def code_only(text):
    """Return text with comments and string/heredoc contents blanked (same length, newlines kept).

    Top-level string quotes are kept so `x = "..."` still reads as an attribute. Template
    interpolations are scanned (they may hold nested strings and braces) and blanked along
    with the string around them, so every brace left in the result is HCL structure.
    """
    out = list(text)
    n = len(text)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    # Frames: top-level "code", "string", or "interp" (code inside ${...}, with its brace depth).
    stack = [["code", 0]]
    i = 0
    while i < n:
        mode, nested = stack[-1][0], len(stack) > 1
        if mode == "string":
            if text[i] == "\\":
                blank(i, i + 2)
                i += 2
            elif text.startswith(("$${", "%%{"), i):  # escaped, literal
                blank(i, i + 3)
                i += 3
            elif text.startswith(("${", "%{"), i):
                blank(i, i + 2)
                stack.append(["interp", 0])
                i += 2
            else:
                if text[i] == '"':
                    stack.pop()
                if text[i] != '"' or len(stack) > 1:
                    blank(i, i + 1)
                i += 1
            continue
        if text[i] == "#" or text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
            continue
        h = _HEREDOC.match(text, i)
        if h:
            t = re.compile(rf"^[ \t]*{re.escape(h.group(1))}[ \t]*$", re.M).search(text, h.end())
            body_end = t.start() if t else n
            end = t.end() if t else n
            blank(i if nested else h.end(), end if nested else body_end)
            i = end
            continue
        c = text[i]
        if c == '"':
            stack.append(["string", 0])
        elif mode == "interp" and c == "{":
            stack[-1][1] += 1
        elif mode == "interp" and c == "}":
            if stack[-1][1] == 0:
                stack.pop()
            else:
                stack[-1][1] -= 1
        if nested:
            blank(i, i + 1)
        i += 1
    return "".join(out)

def blocks(text, header_re):
    """Return each `<header> { ... }` block matching header_re, cut by brace matching.

    Headers inside comments or strings are skipped, and braces are matched on code_only(text),
    so a brace inside a string or comment cannot end a block early. Slices of the original
    text are returned; pass them through code_only() before looking for arguments.
    """
    code = code_only(text)
    out = []
    for m in re.finditer(header_re + r"\s*\{", text):
        if code[m.start()] != text[m.start()] or code[m.end() - 1] != "{":
            continue  # the header, or its brace, sits in a comment or string
        depth = 0
        for j in range(m.end() - 1, len(text)):
            depth += {"{": 1, "}": -1}.get(code[j], 0)
            if depth == 0:
                out.append(text[m.start():j + 1])
                break
        else:
            raise AssertionError(f"unbalanced braces after {m.group(0)!r}")
    return out

def has_attr(code, name):
    """True if code sets argument `name = ...` (a reference like `x.name == y` does not count)."""
    return re.search(rf"(?<![\w.]){re.escape(name)}\s*=(?!=)", code) is not None

def has_block(code, name):
    """True if code holds a nested `name { ... }` block."""
    return re.search(rf"(?<![\w.]){re.escape(name)}\s*\{{", code) is not None

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

def test_blocks_helper_ignores_braces_in_strings_and_comments():
    text = ('resource "a" "x" {\n  description = "close } early"\n  t = "${join("}", [])}"\n'
            '  # stray } here\n  /* and { here */\n  target_tags = ["t"]\n}\n'
            '# resource "a" "c" { commented out }\n')
    [block] = blocks(text, r'resource\s+"a"\s+"\w+"')
    assert block.endswith('target_tags = ["t"]\n}')
    assert has_attr(code_only(block), "target_tags")

def test_code_only_hides_keywords_in_comments_and_strings():
    text = ('resource "google_compute_firewall" "x" {\n'
            '  # log_config { metadata = "INCLUDE_ALL_METADATA" }\n'
            '  // target_service_accounts = [x]\n'
            '  description = "source_ranges = [] and log_config {}"\n'
            '  name = <<EOT\n  target_service_accounts = []\n  EOT\n'
            '  network = google_compute_firewall.y.target_tags == null ? "a" : "b"\n}\n')
    code = code_only(text)
    assert len(code) == len(text)
    for arg in ("target_service_accounts", "source_ranges", "target_tags", "metadata"):
        assert not has_attr(code, arg), arg
    assert not has_block(code, "log_config")
    assert has_attr(code, "description") and has_attr(code, "network")

def test_no_open_internet_source_anywhere():
    offenders = [str(p.relative_to(ROOT)) for p in tf_files() if "0.0.0.0/0" in p.read_text()]
    assert offenders == []

def test_no_iam_resources():
    # Service accounts get no IAM roles (spec section 4.5); no stack manages IAM at all.
    offenders = [str(p.relative_to(ROOT)) for p in tf_files()
                 if re.search(r'resource\s+"google_\w*_iam_', p.read_text())]
    assert offenders == []

def test_firewall_rules_log_and_target_service_accounts():
    # Spec section 3.2: rule 3 (log_config on; target by service account, never by tag) and
    # rule 1 (an ingress rule always declares its source; GCP defaults a missing one to 0.0.0.0/0).
    rules = resource_blocks("google_compute_firewall")
    assert rules, "no google_compute_firewall resources found"
    for path, block in rules:
        where = f"{path.relative_to(ROOT)}: {block.splitlines()[0]}"
        code = code_only(block)
        assert has_block(code, "log_config"), where
        assert has_attr(code, "target_service_accounts"), where
        assert not has_attr(code, "target_tags"), where
        if not re.search(r'(?<![\w.])direction\s*=\s*"EGRESS"', block):
            assert has_attr(code, "source_ranges"), where
            assert not has_attr(code, "source_tags"), where

def test_instances_ignore_boot_image_changes():
    instances = resource_blocks("google_compute_instance")
    assert instances, "no google_compute_instance resources found"
    for path, block in instances:
        assert "ignore_changes=[boot_disk[0].initialize_params[0].image]" in squash(code_only(block)), (
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
