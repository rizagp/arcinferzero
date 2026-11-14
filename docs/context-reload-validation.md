# Keep the configured model's context when it reloads

On main `7ea1d32`, starting an AutoRound catalog model with `--ctx 262144`, then
unloading it through `/v1/server/unload` and naming the same model in the next
chat request, silently reloads it with the catalog's automatic 131072 context.
The model-switch path always set `ctx_auto=true`, including for the startup
model. Its failed-switch recovery path did the same.

The startup model now reuses its startup context policy on reload and rollback.
Other models still select their own automatic context. A startup `--ctx auto`
also remains automatic. No engine, kernel or quantization code changes.

## B70 live validation

Native Linux, B70 32 GB, i7-14700K, 96 GB RAM; AutoRound INT4 routed-expert hybrid,
UD-Q4_K_XL non-expert tensors, 64 GiB expert RAM, MTP off, chunk 2048, medium
reasoning, temperature zero. No `--precise` or `ARC_FN_PRECISE`: current main
automatically selects precise mode for AR4. The validation build includes the
Linux loading fixes in PR #24 and the separately submitted revision guard.

| Actual server check | Before | After |
|---|---|---|
| Startup with `--ctx 262144` | 262144 | 262144 |
| Explicit unload, reload the same catalog model by alias | 131072 | 262144 |
| Select a metadata-only model with missing hparams | — | Expected HTTP 500; startup model restored at 262144 |

Two identical arithmetic requests before and after unload/reload produce the
same complete choice (including reasoning) and completion-token count. This is
an API/state check, not a before/after dump of every logit. The process-lifetime
precision counter also returns to zero after unloading in the independent
actual-Engine dry test; automatic precise mode is logged again on live reload.

Both runz and serverz were rebuilt before the live checks. No inference workload
ran alongside the tests, and the original Pi service was restored afterward.
