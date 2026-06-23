# inspect-nflreadpy — Introspect nflreadpy

```bash
python3 << 'PY'
import inspect
import nflreadpy

print("nflreadpy:", nflreadpy)
print("version:", getattr(nflreadpy, "__version__", "unknown"))

public = [name for name in dir(nflreadpy) if not name.startswith("_")]
print("public names:")
for name in public:
    print(" -", name)

print("\nLikely data-loading functions:")
for name in public:
    lower = name.lower()
    if any(token in lower for token in ["schedule", "game", "load", "import", "read"]):
        obj = getattr(nflreadpy, name)
        print("\n---", name, "---")
        print(obj)
        try:
            print("signature:", inspect.signature(obj))
        except Exception as exc:
            print("signature unavailable:", exc)
        doc = getattr(obj, "__doc__", None)
        if doc:
            print("doc:", doc[:1200])
PY
```
