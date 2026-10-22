# Distributed logging

`jigsaw.log` is a rank-aware wrapper over the stdlib `logging` module. `init_cuda()`
calls `log.configure_logging()` for you; call it yourself (once, after the launcher
env is known) only if you want to log before `init_cuda` or override the defaults.

```python
from jigsaw import log

log.info("world_size=%d", world)  # emitted by the print rank only (default: rank 0)
log.info_all("process group ready")  # every rank
log.at(2).debug("q shard %s", q.shape)  # only rank 2 (accepts an int, a list, or "all")
```

Every record is stamped with `[rank/world]` and an elapsed-time marker. Other helpers:

- `log.assert_all_agree("seed", seed)`: collective check that a value matches across ranks;
  calls `log.die` on a mismatch unless `fatal=False`.
- `with log.timed("forward", barrier=True, cuda_sync=True): ...`: time a block, log it from the print rank.
- `log.log_ranks("loss=%.4f", loss, level=logging.INFO)`: gather one line per rank and print
  them in order from rank 0. The default level is `DEBUG`, hidden at the default log level.
- `log.die("bad shard %s", shape)`: log on the calling rank, tear down the process group, exit (no hang).
- `configure_logging()` also installs `faulthandler` (`kill -USR1 <pid>` dumps stacks) and a rank-aware
  `excepthook` so a crashing rank fails the job instead of deadlocking its peers.

Configured via arguments to `configure_logging()` or environment variables (arguments win):
`JIGSAW_LOG_LEVEL` (default `INFO`), `JIGSAW_LOG_RANK` (default print rank `0`),
`JIGSAW_LOG_DIR` (write per-rank `rank_<NNNN>.log` files; default off),
`JIGSAW_LOG_FORMAT` (`json` for one JSON object per line).
