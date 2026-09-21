# Examples

Each example uses the configured Laya model and can be run from the project root:

```bash
uv run python examples/support_triage.py
uv run python examples/project_routing.py
uv run python examples/incident_grouping.py
uv run python examples/command_palette.py
uv run python examples/timing.py
```

- `basic.py` shows the smallest possible semantic function and its full `Decision`.
- `support_triage.py` evaluates three questions in one model pass with `semfn.gather`.
- `project_routing.py` turns a runtime list into choices and returns the original object.
- `incident_grouping.py` combines semantic comparison with ordinary application logic.
- `command_palette.py` selects an executable object while rejecting uncertain matches.
- `timing.py` measures model loading, first inference, and a warm inference.

The first run downloads the configured model. Later processes still check the local
cache, but semfn hides that progress display by default. Pass `quiet=False` to
`semfn.configure()` if you want to see it.

The examples need no configuration because semfn creates the default runtime on
the first call. Use `semfn.configure()` only to override its defaults or to get a
runtime for explicit warmup.

Within one process, the configured runtime loads the model only once. Call
`await runtime.warmup()` during application startup to pay the loading cost before
the first user request.
