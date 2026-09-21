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

## Evals

A semantic function is only as good as its answers on your data. Attach labeled
cases to it and measure:

```python
same_incident.eval(
    [
        semfn.case(error1, error2, expected=True),
        semfn.case(error1, error3, expected=False),
    ]
)
```

```bash
semfn eval            # every *.py under evals/
semfn eval -v path/   # also list the failed cases
```

```text
same_incident     91/100
classify_ticket   96/100
project_for       73/100  ← below 80%
```

The exit code is 1 when a function falls below `--min-accuracy` (default 0.8),
so the command works as a CI gate. `--backend` and `--model` compare backends on
the same cases. Cases run through the function's confidence policy, so the
report describes what callers actually get back. `await fn.eval([...])` runs the
cases in-process and returns an `EvalReport`.

The suites in [`evals`](evals) cover five small applications.

## Development

```bash
just ci  # ruff format --check, ruff check, ty check, pytest
```
