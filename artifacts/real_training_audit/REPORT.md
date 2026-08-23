# Real GRPO training audit (Qwen3-4B, autodl2, 2026-08-23)

- data dir: artifacts\real_training

## Rollout policy sync (static_rollout offline signal)
- TRL sync calls observed: 398
- acked: 398 | unacked: 0
- committed optimizer steps: 1
- verdict: CLEAR (all sync calls acked)

## Policy traceability
- model: Qwen/Qwen3-4B | policy v1 (parent v0)
- weight shards: 5 | total bytes: 17,645,742,424
- traceability ok: True

## Findings
- static_rollout signal clear: 398/398 sync calls acked

## Honesty notes
- Real artifacts pulled with the project owner's explicit authorization; the smoke loop was a real Qwen3-4B GRPO run (398 parameter sync calls, 1 optimizer step committed).
- The audit covers offline-detectable signals from the manifests; blob contents are Guard's binary event store (decoding stays Guard-side, design 25).