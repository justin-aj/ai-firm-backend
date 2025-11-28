# GPT-OSS Guide (Archived)

This document contains an archive of the GPT-OSS integration that existed in the
project prior to its deprecation in favor of vLLM. The code in `deprecated/gpt_oss`
directory is intentionally left intact for reference and historical purposes.

## Rationale
- GPT-OSS was deprecated due to project requirements that favor vLLM and to
  reduce the complexity of maintaining two inference backends.

## Where to find the old code
- clients/gpt_oss_client.py -> archived in deprecated/gpt_oss/clients
- routes/gpt_oss.py -> archived in deprecated/gpt_oss/routes
- tests/test_gpt_oss.py -> archived in deprecated/gpt_oss/tests

## Note
This archive is only for historical reference. Do not restore or use this code in
active deployments.
