"""#6: the README credit claims a re-render on every change only for repos whose CI calls the reusable render workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint.readme import embed, renders_in_ci

CLAIM = "on every change to the model"


def _repo(tmp_path: Path, workflow: str | None, name: str = "publish.yml", git_file: bool = False) -> Path:
    """A fake repo root: .git (a directory, or a file as in a worktree), optionally one workflow, and a README one level down."""
    (tmp_path / ".git").write_text("gitdir: elsewhere\n") if git_file else (tmp_path / ".git").mkdir()
    if workflow is not None:
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / name).write_text(workflow)
    (tmp_path / "part").mkdir()
    return tmp_path / "part" / "README.md"


# The callers' real lines (tag-clip, brackets, gen-o-lantern publish.yml; this repo's publish.yml), plus a list-item form.
CALLS = [
    "jobs:\n  render:\n    uses: csmarshall/common-blueprint-generator/.github/workflows/render.yml@v1\n",
    "jobs:\n  render:\n    uses: ./.github/workflows/render.yml\n",
    "jobs:\n  render:\n    uses: 'csmarshall/common-blueprint-generator/.github/workflows/render.yml@main'\n",
]
# Not a call: a comment quoting the line (as release.yml does), another project's render.yml, a repo with no render workflow.
NOT_CALLS = [
    "#       uses: csmarshall/common-blueprint-generator/.github/workflows/render.yml@v1\n",
    "jobs:\n  r:\n    uses: someone/else/.github/workflows/render.yml@v1\n",
    "jobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: make blueprint\n",
]


@pytest.mark.parametrize("workflow", CALLS)
def test_ci_repo_keeps_the_claim(tmp_path: Path, workflow: str) -> None:
    readme = _repo(tmp_path, workflow)
    assert renders_in_ci(readme)
    assert CLAIM in embed(readme, readme.parent / "a.svg", "A", "a.step")


@pytest.mark.parametrize("workflow", [*NOT_CALLS, None])
def test_local_repo_drops_the_claim(tmp_path: Path, workflow: str | None) -> None:
    readme = _repo(tmp_path, workflow)
    assert not renders_in_ci(readme)
    text = embed(readme, readme.parent / "a.svg", "A", "a.step")
    assert CLAIM not in text
    assert "by [common-blueprint-generator](https://github.com/csmarshall/common-blueprint-generator) · [Buy me a coffee]" in text


def test_worktree_git_file_and_yaml_extension(tmp_path: Path) -> None:
    assert renders_in_ci(_repo(tmp_path, CALLS[0], name="blueprint.yaml", git_file=True))


def test_environment_does_not_decide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running under Actions must not add the claim to a locally rendered repo, or the README would flip between renders."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert not renders_in_ci(_repo(tmp_path, NOT_CALLS[2]))


def test_explicit_override(tmp_path: Path) -> None:
    readme = _repo(tmp_path, None)
    assert CLAIM in embed(readme, readme.parent / "a.svg", "A", "a.step", ci=True)
    assert CLAIM not in embed(readme, readme.parent / "a.svg", "A", "a.step", ci=False)


def test_this_repo_renders_in_ci() -> None:
    """This repo's own README keeps the claim: publish.yml calls render.yml, so the example's embed must not change."""
    assert renders_in_ci(Path(__file__).resolve().parents[1] / "README.md")
