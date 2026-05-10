"""Build .docs-sync-digest/ for the krkn-ai repo.

krkn-ai is a Python project with a Click-based CLI in `krkn_ai/cli/cmd.py`.
The documented surface is the set of CLI commands (`run`, `monitor`,
`discover`) and their `@click.option` flags.

Output:
  - llms.txt        — index (one line per CLI command)
  - llms-full.txt   — full structured detail with option tables
  - digest.sha      — sha256 of source files for cache invalidation

Format mirrors krkn-hub's digest so the website-side extractor can share
parsing logic. The `## scenario:` heading is the generic "entity" anchor;
for krkn-ai it represents a CLI command rather than a chaos scenario.

Pure deterministic Python AST walk. No LLM. Bit-identical output for
same input.

Run from the krkn-ai repo root:
    python .docs-sync/build_upstream_digest.py
"""
import argparse
import ast
import hashlib
import sys
from pathlib import Path
from typing import Iterable


# We extract from these files. List is explicit, not glob, because adding
# a new source file is itself a documentation event — the maintainer
# should decide whether to surface it to the docs.
_CLI_FILES = [
    Path("krkn_ai/cli/cmd.py"),
]


def _str_literal(node: ast.expr) -> str:
    """Return the string value of a literal-or-call node, or '' if not extractable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        # f-string — concatenate the literal parts; skip expressions
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        return "".join(parts)
    return ""


def _is_click_decorator(deco: ast.expr, name: str) -> bool:
    """True if `deco` is `@click.<name>(...)` or `@main.<name>(...)`."""
    call = deco.func if isinstance(deco, ast.Call) else deco
    if not isinstance(call, ast.Attribute):
        return False
    return call.attr == name


def _is_main_command(deco: ast.expr) -> bool:
    """True if `deco` is `@main.command(...)`."""
    if not isinstance(deco, ast.Call):
        return False
    if not isinstance(deco.func, ast.Attribute):
        return False
    return deco.func.attr == "command" and (
        isinstance(deco.func.value, ast.Name) and deco.func.value.id == "main"
    )


def _is_click_option(deco: ast.expr) -> bool:
    """True if `deco` is `@click.option(...)`."""
    if not isinstance(deco, ast.Call):
        return False
    if not isinstance(deco.func, ast.Attribute):
        return False
    return deco.func.attr == "option" and (
        isinstance(deco.func.value, ast.Name) and deco.func.value.id == "click"
    )


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _parse_click_option(call: ast.Call) -> dict:
    """Extract one option's metadata from `@click.option(...)`."""
    # Positional args are the flag forms: "--namespace", "-n", etc.
    flag_args = [
        _str_literal(a) for a in call.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    ]
    flag_args = [f for f in flag_args if f]

    # The "primary" name is the longest `--<name>` flag; param name is the
    # one Click would derive (longest --foo-bar → foo_bar). The kebab-case
    # form is the user-visible CLI form.
    long_flags = sorted((f for f in flag_args if f.startswith("--")), key=len, reverse=True)
    primary = long_flags[0] if long_flags else (flag_args[0] if flag_args else "")
    kebab = primary[2:] if primary.startswith("--") else primary
    variable = kebab.replace("-", "_").upper()

    help_node = _kwarg(call, "help")
    default_node = _kwarg(call, "default")
    type_node = _kwarg(call, "type")
    required_node = _kwarg(call, "required")
    is_flag_node = _kwarg(call, "is_flag")

    description = _str_literal(help_node) if help_node else ""

    # Default formatting
    if default_node is None:
        default_str = ""
    elif isinstance(default_node, ast.Constant):
        if default_node.value is None:
            default_str = ""
        elif isinstance(default_node.value, bool):
            default_str = "true" if default_node.value else "false"
        else:
            default_str = str(default_node.value)
    elif isinstance(default_node, ast.Call):
        # e.g. `default=os.getenv("KUBECONFIG", None)` — record as "(dynamic)"
        default_str = "(dynamic)"
    elif isinstance(default_node, ast.List):
        default_str = "[]"  # most empty-list defaults
    else:
        default_str = ""

    # Type formatting — recognize a few common Click idioms; default to "string"
    if is_flag_node and isinstance(is_flag_node, ast.Constant) and is_flag_node.value:
        type_str = "bool"
    elif type_node is None:
        type_str = "string"
    elif isinstance(type_node, ast.Name):
        # type=int / type=str
        type_str = {"int": "number", "str": "string", "float": "number",
                    "bool": "bool"}.get(type_node.id, type_node.id)
    elif isinstance(type_node, ast.Call) and isinstance(type_node.func, ast.Attribute):
        # type=click.Choice([...]) etc.
        type_str = type_node.func.attr.lower()
    else:
        type_str = "string"

    required = False
    if required_node is not None and isinstance(required_node, ast.Constant):
        required = bool(required_node.value)

    return {
        "name": kebab,
        "variable": variable,
        "type": type_str,
        "default": default_str,
        "required": required,
        "description": description,
    }


def _extract_command_metadata(func: ast.FunctionDef) -> dict | None:
    """Given a FunctionDef, return command metadata if it's @main.command-decorated."""
    is_command = False
    command_help = ""
    options: list[dict] = []

    for deco in func.decorator_list:
        if _is_main_command(deco):
            is_command = True
            # Extract help= kwarg or the first positional string arg
            help_node = _kwarg(deco, "help") if isinstance(deco, ast.Call) else None
            if help_node is not None:
                command_help = _str_literal(help_node)
        elif _is_click_option(deco):
            options.append(_parse_click_option(deco))

    if not is_command:
        return None

    # Click decorators stack in source order but apply bottom-up. The list
    # we build here is in declaration order — reverse so the digest reads
    # top-to-bottom in the same order a user sees in --help.
    options.reverse()

    return {
        "name": func.name,
        "description": command_help,
        "options": options,
    }


def extract_cli_commands(repo_root: Path) -> list[dict]:
    """Walk the CLI source files; return one dict per command."""
    commands: list[dict] = []
    for rel in _CLI_FILES:
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                meta = _extract_command_metadata(node)
                if meta is not None:
                    commands.append(meta)
    return commands


def render_llms_txt(commands: list[dict], repo_name: str) -> str:
    """One-line-per-command index."""
    lines = [
        f"# {repo_name}",
        "",
        "> Auto-generated by .docs-sync/build_upstream_digest.py.",
        "> Do not edit by hand. Used by the docs-sync agent to detect upstream changes.",
        "",
        "## CLI commands",
        "",
    ]
    for c in sorted(commands, key=lambda c: c["name"]):
        opt_count = len(c.get("options", []))
        lines.append(
            f"- {c['name']} ({opt_count} option{'s' if opt_count != 1 else ''})"
        )
    return "\n".join(lines) + "\n"


def render_llms_full_txt(commands: list[dict], repo_name: str) -> str:
    """Per-entity detail file. Reuses krkn-hub's section format so the
    website-side extractor can share parsing logic. For krkn-ai, each
    "## scenario:" is actually a CLI command."""
    lines = [
        f"# {repo_name} — full entity details",
        "",
        "> Auto-generated. Source of truth for the krkn-ai CLI surface.",
        "",
    ]
    for c in sorted(commands, key=lambda c: c["name"]):
        # Use krkn-hub's "scenario:" heading for cross-compat with the
        # website's extractor. scenario_type carries the entity kind.
        lines.append(f"## scenario: {c['name']}")
        lines.append("scenario_type: cli_command")
        if c.get("description"):
            lines.append(f"description: {c['description']}")
        lines.append("")

        opts = c.get("options", [])
        if opts:
            lines.append("### parameters")
            lines.append("")
            lines.append("| name | variable | type | default | required | description |")
            lines.append("| ---- | -------- | ---- | ------- | -------- | ----------- |")

            def cell(v):
                return str(v).replace("|", "\\|").replace("\n", " ")

            for o in opts:
                lines.append(
                    f"| {cell(o['name'])} "
                    f"| {cell(o['variable'])} "
                    f"| {cell(o['type'])} "
                    f"| {cell(o['default'])} "
                    f"| {str(o['required']).lower()} "
                    f"| {cell(o['description'])} |"
                )
            lines.append("")
        else:
            lines.append("(no documented options)")
            lines.append("")
    return "\n".join(lines) + "\n"


def compute_digest_sha(repo_root: Path) -> str:
    """sha256 of all source files we extracted from."""
    h = hashlib.sha256()
    for rel in sorted(_CLI_FILES):
        path = repo_root / rel
        if not path.is_file():
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def build_upstream_digest(
    repo_root: Path, output_dir: Path, repo_name: str,
) -> dict:
    commands = extract_cli_commands(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    llms = render_llms_txt(commands, repo_name)
    full = render_llms_full_txt(commands, repo_name)
    sha = compute_digest_sha(repo_root)

    (output_dir / "llms.txt").write_text(llms, encoding="utf-8")
    (output_dir / "llms-full.txt").write_text(full, encoding="utf-8")
    (output_dir / "digest.sha").write_text(sha + "\n", encoding="utf-8")

    return {"command_count": len(commands), "digest_sha": sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path(".docs-sync-digest"))
    parser.add_argument("--repo-name", default="krkn-ai")
    args = parser.parse_args(argv)

    if not args.repo_root.is_dir():
        print(f"error: repo root not found: {args.repo_root}", file=sys.stderr)
        return 2

    result = build_upstream_digest(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        repo_name=args.repo_name,
    )
    print(
        f"Wrote krkn-ai digest: {result['command_count']} CLI commands, "
        f"sha={result['digest_sha'][:8]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
