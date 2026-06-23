# discover — Inspect repo state

```
pwd
git status --short || true
find . -maxdepth 4 -type f | sort | sed 's#^./##' | head -250
test -f pyproject.toml && cat pyproject.toml || true
test -f Makefile && cat Makefile || true
test -f AGENTS.md && cat AGENTS.md || true
```
