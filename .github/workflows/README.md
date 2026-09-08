# GitHub Actions — temporarily disabled

**Status: DISABLED as of 2026-09-08. Keep disabled until the GitHub Actions
issue is resolved. Do not re-enable before then.**

All GitHub Actions workflows in this directory have been renamed from
`*.yml` to `*.yml.disabled`. GitHub Actions only picks up files ending in
`.yml` / `.yaml` inside `.github/workflows/`, so none of these workflows
will trigger on push, tag, or pull request until they are renamed back.

## Reason

GitHub Actions has an open issue affecting this repository. While the
issue is unresolved, CI coverage is provided by the GitLab mirror pipeline
instead (see the `gitlab` remote and `.gitlab-ci.yml`).

## Disabled workflows

| Workflow file | Triggers | Purpose |
| --- | --- | --- |
| `build-page-docs.yml` | `v*.*.*` tags, manual | Sphinx docs build + deploy to GitHub Pages |
| `publish-posix-to-pypi.yml` | `v*.*.*` tags, manual | Linux cibuildwheel wheels → PyPI |
| `publish-nt-to-pypi.yml` | `v*.*.*` tags, manual | Windows cibuildwheel wheels → PyPI |

## How to re-enable

Only once the GitHub Actions issue is resolved, rename the files back and
push to `origin` (GitHub):

```sh
git mv .github/workflows/build-page-docs.yml.disabled .github/workflows/build-page-docs.yml
git mv .github/workflows/publish-posix-to-pypi.yml.disabled .github/workflows/publish-posix-to-pypi.yml
git mv .github/workflows/publish-nt-to-pypi.yml.disabled .github/workflows/publish-nt-to-pypi.yml
git commit -m "ci: re-enable GitHub Actions workflows"
git push origin main
```

Then restore the GitHub Actions badges in `README.md`.
