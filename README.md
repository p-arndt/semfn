# semfn

Turn ordinary typed Python functions into semantic, probabilistic functions.

```python
from typing import Literal

import semfn
from semfn import semantic


@semantic
def classify(text: str) -> Literal["bug", "feature", "question"]:
    """What kind of request is this?"""


kind = await classify("Login redirects forever")
decision = await classify.result("Login redirects forever")
print(kind, decision.confidence, decision.distribution)
```

The initial API supports `bool`, `Literal`, string enums, optional results,
dynamic object selection from `list[T]`, ordinal `Score[...]` results,
confidence policies, and explicit batching.

```python
async with semantic.batch() as pending:
    classify(text)
    is_blocking(text)

kind, blocking = await pending.resolve()
```

More complete, runnable examples live in [`examples`](examples/README.md).

The default Laya model is configured automatically on the first call. Call
`configure()` only when you want a different model, backend, or runtime option:

```python
semfn.configure(model="my-org/my-laya-model")
```

To load the model during application startup and keep the first user request fast:

```python
runtime = semfn.configure()
await runtime.warmup()
```

The Laya download/cache progress display is hidden by default. Use
`semfn.configure(quiet=False)` to show it.
