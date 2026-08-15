# Provenance

## Seed source

This repository was created as an isolated working copy of the CRISP **reimplementation**, not the original reference implementation.

- **Copied from:** `../crisp-reimpl/crisp-reimpl/` (relative to this project)
- **Workspace source path:** `D:\IOAI\AIYGO\CRISP-main\crisp-reimpl\crisp-reimpl`
- **Copy timestamp:** 2026-07-29T12:38:27+0530 (IST)
- **Copy policy:** excluded `.git`, virtual environments, bytecode/cache directories, experiment outputs, W&B artifacts, and generated Python bytecode.
- **Initial manifest:** 50 source files, SHA-256 listed below.

## Why this source was chosen

The seed project is the independently structured CRISP reimplementation. Its README documents response-token-aligned scoring, detached teacher KL targets, symmetric PC-Grad calculated from original gradients, and unit coverage for pure-Python components. It also honestly states that model/GPU paths require a real smoke test before expensive runs are trusted.

The separate reference project at `../CRISP-main/` was deliberately not copied or modified. It is not the implementation base for this project.

## Source status caveat

The copied README reports that static/pure-Python validation and a small real smoke run had found and fixed several issues, including EOS-token handling and a Transformers dtype-argument compatibility issue. It does **not** establish benchmark results or prove all GPU/model configurations work. Any validation performed in this derived project must be recorded independently.

## SHA-256 source manifest

```text
1e419870d3fd8d86e96a3f84c1061640e2f7ed2f085d719e705c4e88b1a3f813  config/crisp_7b.yaml
7c60de9ce2dc874a73f6e7aff738ce689c2cc6a55bc29ebff57a1f8f6daf48d8  config/crisp_pilot.yaml
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/baselines/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/baselines/grpo/__init__.py
3013cb5f4e1ed6e66352d0f99d03b44f28cf1dbc41f2ff02964a1bc52274c49d  crisp/baselines/grpo/training_loop.py
2831ccc61c3c4961b2173ef4198c381bf0468a1964f0272c624252f2df7462cd  crisp/baselines/grpo/training_step.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/baselines/opsd/__init__.py
5fa4843a757ac1e3aa3654cca522f55e46ebfb4fbb3473ce3bafd7d6d74904fc  crisp/baselines/opsd/training_loop.py
9ed7692564cd901f16521b6c2aa7db40f8c0dd6b7104a2bdc4b4f5ea229547c5  crisp/baselines/opsd/training_step.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/baselines/reinforce_pp/__init__.py
694c06aab0d49c0662f59c645374152c29a457b86068f3e8212fc2387a8f231f  crisp/baselines/reinforce_pp/training_loop.py
eea7d674659cb6950416121b33f0c485ab8d7aa649082870e0ab2ff4c71b2b38  crisp/baselines/reinforce_pp/training_step.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/data/__init__.py
1cbddb0455dc898e52afc352ebe3dbc5a176050dfa18df74a050cd999baaf42e  crisp/data/build_dataset.py
cbda47de5d31c29bec11d1de46face48ea3c0d76e9e91c4407267c53d0ecb344  crisp/data/reward.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/eval/__init__.py
053c6406e5decc5296a8544c3a432a86ca4c6cc101c7a93dd587fc117263852d  crisp/eval/metrics.py
f826c085993155a4109a3cbae9ce00925013e9612a36c2d159ec2270e992b950  crisp/eval/run_eval.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/model/__init__.py
57f0e3b5276ab634e4df24c187edcb6c9af33267dbf4c504d7ea89e87a7ca19c  crisp/model/load.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/training/__init__.py
835804759e3a53c5d89782ddaabfc38c9fa2f6107a262bcfb3f2b42011c979ec  crisp/training/common_loop.py
cd8985e44b599fdec8182d619472fedb1612512332c151070caa455ac6c86d03  crisp/training/crisp_step.py
3a5b7db465e7b825d64f3472a35ae63f1c7f385d255ad16042b2b5c6b3ce0f76  crisp/training/eos_utils.py
cc102bb6d862f82579b1548cc5a28af4d8c98d8f69b792c3473565edba17c3bb  crisp/training/grad_utils.py
a4ac38cc9d505a38b9c63707639bb39212bc61452901412cb5c8bbb8f1a85a67  crisp/training/logprobs.py
bf3e32d4fd6ddc0b9566e9c7ce3631dc775e4b9ac973df0f74201c14f23cfc05  crisp/training/pcgrad.py
62c2262702efe2f33657b231e6ca91138eabcb63079d1205401ac5c828f86e71  crisp/training/rollout.py
433dde980e413f8fad1024fa0c5364d7015a12848013daef4dd23fd173b27ca5  crisp/training/schedules.py
562a6293fcb3f21d2ee461000b044d8d2b4e5a0a9a7f404cbb81456319ce5800  crisp/training/train.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  crisp/utils/__init__.py
c412d866539de47a6ca500fc94a8148456aa5cc2777c7352f35c4e96ecd0d096  crisp/utils/config.py
ef017a1d90d241428f6357c455447b06ed67561ca807afa91dfd2483b5ac4b9b  crisp/utils/device.py
dbd7e90dd2a0165faa1ca07e25b0fb6981d400e99fb19d2bf40d6a675966a464  crisp/utils/logging_utils.py
62609b8e2196d59cb784a8dea4fcf3c3c145101e566ebd90a17cc17c75a1f33b  crisp/utils/seeding.py
ad9bd1e8fa786310f483d99eff276c5b116299df1284462b4030e74094f3e834  predownload.py
c2f77ae1533e5fc2f16e9d71a8ff5bda422150fcf93e22beec7d8f93c7d7d972  requirements.txt
09468b7f8fb2cf01f0e6548cec7f093a01bd867a1d37601029fa891164b71947  sweep.sh
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/__init__.py
3eef44a0e116b48d0a85dce1c1e5f9dd5000588e53bbab431e7fcd54e0bc4bcd  tests/test_build_dataset.py
313106ada335a2e2a7cd9c806bf8d821f5d65c05b497ed51c16f427a60c13127  tests/test_config.py
da6de8eb1befa1fceb565cab7d8f961c8d3a95644bd9f35bc2bd8edbe3dc5c66  tests/test_eos_utils.py
7742c32e9a14ceee4d0a7343eb37c13bb97d4c434dca9c0351d8514392a67236  tests/test_logprobs.py
19d84ff73f15fc9134b98d75caa6305e17553fa1c0f269167b6d4fe7aea71446  tests/test_metrics.py
653f467151324ae8023f18b536d8a4200557477e0359c7d5fc9f0504acae21a2  tests/test_pcgrad.py
5faddeef4a6ea2b9323470961a5c9ebc2919330d68f730899d0a581353bb242b  tests/test_reward.py
cbd831fbfbec77f5162c9cb51d0fc76386df06b409a523ca62fe7d5d717c9562  tests/test_schedules.py
c7eeaea7d01ec7a3efc06bb773c3edb36fada691a7e676ae87c3f7e297bd535a  tests/test_train_launcher.py
ce6b26551fa6bab17c18b8aae3805ec8d68e9141fa11c70295fb0e37b507f477  train_launcher.py
```

## Initial verification

The unmodified seed was committed as `e6228dd` before feature work. The following checks were run in this derived repository:

```text
python -m compileall -q .
compileall: PASS

uv run --with sympy --with pyyaml --with pytest pytest tests -q
113 passed, 17 skipped, 6 subtests passed
```

The skipped tests are Torch-dependent unit tests; no GPU/model training claim is implied by this result. A file-level comparison also confirmed that the copied legacy `crisp/training/crisp_step.py` and all legacy baseline source files remained identical to the seed after Task 2 configuration work.
