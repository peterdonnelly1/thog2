# THOG2 Stage 6 Controlled Pilot Analysis

Protocol: `1a7c66a02e5480c862f9c13d7cc3231eafa3b54c688c08093f5744bc6c16d490`

## Control validation

All completed runs used the same protocol, completed updates, consumed tokens, training batch trace, validation sample, and evaluation updates.

## Resource and final-loss summary

| Run | Persistent parameters | Reduction vs dense | Peak allocated GiB | Tokens/s | Final validation loss |
|---|---:|---:|---:|---:|---:|
| Dense | 549,158,400 | 0.00% | 10.069 | 3693.7 | 6.843831 |
| Sheet Q64 | 48,282,112 | 91.21% | 1.060 | 1232.4 | 6.470158 |
| Sheet Q128 | 57,732,608 | 89.49% | 1.266 | 1207.7 | 6.454906 |
| Sheet Q256 | 76,633,600 | 86.05% | 1.669 | 1026.6 | 6.478035 |

## Scientific classification

**Pending review.** This generated analysis validates controls and tabulates measurements; it does not choose the final Stage 6 classification automatically.

Allowed final classifications:

- viable for further study;
- viable only at weak compression;
- inconclusive;
- not viable under the tested design.

## Scope limitation

This pilot is matched at `L72/H12/D768/C256`. It does not establish a matched dense comparison at L144.

## Trace digests

```json
{
  "dense": {
    "completed_updates": 250,
    "consumed_tokens": 1024000,
    "evaluation_updates": [
      0,
      25,
      50,
      75,
      100,
      125,
      150,
      175,
      200,
      225,
      250
    ],
    "protocol_match": true,
    "training_trace_sha256": "66c6663db2820cdc2648efbd8ac15ee1082b80800b5ded34694c08a30d1638b0",
    "validation_trace_sha256": "b0973c3b94bcf1d8662770861ad0a3cc69f4f2a42b53775bf08351e38d18cef6"
  },
  "q128": {
    "completed_updates": 250,
    "consumed_tokens": 1024000,
    "evaluation_updates": [
      0,
      25,
      50,
      75,
      100,
      125,
      150,
      175,
      200,
      225,
      250
    ],
    "protocol_match": true,
    "training_trace_sha256": "66c6663db2820cdc2648efbd8ac15ee1082b80800b5ded34694c08a30d1638b0",
    "validation_trace_sha256": "b0973c3b94bcf1d8662770861ad0a3cc69f4f2a42b53775bf08351e38d18cef6"
  },
  "q256": {
    "completed_updates": 250,
    "consumed_tokens": 1024000,
    "evaluation_updates": [
      0,
      25,
      50,
      75,
      100,
      125,
      150,
      175,
      200,
      225,
      250
    ],
    "protocol_match": true,
    "training_trace_sha256": "66c6663db2820cdc2648efbd8ac15ee1082b80800b5ded34694c08a30d1638b0",
    "validation_trace_sha256": "b0973c3b94bcf1d8662770861ad0a3cc69f4f2a42b53775bf08351e38d18cef6"
  },
  "q64": {
    "completed_updates": 250,
    "consumed_tokens": 1024000,
    "evaluation_updates": [
      0,
      25,
      50,
      75,
      100,
      125,
      150,
      175,
      200,
      225,
      250
    ],
    "protocol_match": true,
    "training_trace_sha256": "66c6663db2820cdc2648efbd8ac15ee1082b80800b5ded34694c08a30d1638b0",
    "validation_trace_sha256": "b0973c3b94bcf1d8662770861ad0a3cc69f4f2a42b53775bf08351e38d18cef6"
  }
}
```
