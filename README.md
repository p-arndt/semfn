# semfn

Turn ordinary typed Python functions into semantic, probabilistic functions.

```python
from typing import Literal

import semfn


@semfn.semantic
def classify(text: str) -> Literal["bug", "feature", "question"]:
    """What kind of request is this?"""


kind = await classify("Login redirects forever")
decision = await classify.result("Login redirects forever")
print(kind, decision.confidence, decision.distribution)
```

The docstring is the question, the arguments are the state, and the return
annotation defines the possible answers. The body never runs. Signatures are
validated when the function is decorated, and calls keep their parameter types.

| Return annotation | Answer |
| --- | --- |
| `bool` | yes or no |
| `Literal[...]`, an `Enum` | one of the listed values |
| `Score[Literal["low", "high"]]` | one of the ordered levels, typed as the `Literal` |
| `T` with one `list[T]`, `Sequence[T]` or `tuple[T, ...]` parameter | one of the objects passed in |
| any of these `\| None` (except `Score`) | adds "none" as an answer |

`Decision.confidence` is always the probability of the selected answer, so a
threshold means the same for every return type:

```python
@semfn.semantic(min_confidence=0.8)  # raises UncertainDecision below 0.8
def is_blocking(text: str) -> bool:
    """Is the user unable to continue?"""


@semfn.semantic(min_confidence=0.8, uncertain=None)  # typed as bool | None
def is_spam(text: str) -> bool:
    """Is this spam?"""
```

Questions about the same arguments share one model pass with `gather`, which is
typed like `asyncio.gather`:

```python
kind, blocking = await semfn.gather(classify(text), is_blocking(text))
```

Methods work too; the instance becomes part of the state.

More complete, runnable examples live in [`examples`](examples/README.md).

The default Laya model is configured automatically on the first call. Call
`configure()` only when you want a different model, backend, or runtime option:

```python
semfn.configure(model="my-org/my-laya-model")
```

The same functions can use TypeSafe's hosted Jev model without changing their
definitions:

```bash
pip install "semfn[jev]"
export TYPESAFE_API_KEY=...
```

```python
semfn.configure(backend="jev")
```

Pass `model="..."` to select a different model for either backend. Jev receives
the function arguments as state and uses the same `noul`, `choice`, and `score`
questions as Laya.

To load the model during application startup and keep the first user request fast:

```python
runtime = semfn.configure()
await runtime.warmup()
```

The Laya download/cache progress display is hidden by default. Use
`semfn.configure(quiet=False)` to show it.

In tests, or to route one request elsewhere, scope a backend instead of changing
the global one. A backend is any object with an `evaluate(state, questions)`
method, see `semfn.Backend`:

```python
with semfn.using(FakeBackend()):
    assert await classify("Login loops") == "bug"
```

`@semfn.semantic(backend=...)` pins a single function to its own backend.

## Development

```bash
just ci  # ruff format --check, ruff check, ty check, pytest
```
