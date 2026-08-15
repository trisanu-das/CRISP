#!/usr/bin/env python
"""
Predownload a model's weights/tokenizer before training touches them.

This exists specifically to decouple "is the download working" from "is
training working" -- see the README's "Model download is slow or hangs"
section. Run it standalone first:

    python predownload.py Qwen/Qwen2.5-0.5B-Instruct

If it hangs or stalls (progress bar stops updating, no error, nothing
happens), that's a known class of issue with HuggingFace Hub's newer
chunk-based transfer backend ("Xet") on some networks -- try:

    python predownload.py Qwen/Qwen2.5-0.5B-Instruct --disable-xet

Safe to re-run: `snapshot_download` resumes from whatever's already been
fetched rather than starting over. Once this succeeds, set
`model.local_files_only: true` in your config so training never attempts a
network call at all.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_name", help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--disable-xet", action="store_true",
        help="Fall back to plain HTTP transfer instead of HF Hub's chunk-based Xet backend. "
             "Slower, but avoids a known class of hang on some networks/firewalls.",
    )
    parser.add_argument(
        "--high-performance", action="store_true",
        help="Opt into Xet's high-throughput mode (more concurrency/RAM). Only useful if the "
             "download is merely slow rather than hanging -- don't combine with --disable-xet.",
    )
    parser.add_argument("--revision", default=None, help="Specific revision/branch/commit to fetch")
    args = parser.parse_args()

    if args.disable_xet and args.high_performance:
        parser.error("--disable-xet and --high-performance are mutually exclusive")

    # These must be set before huggingface_hub (and its Xet transfer backend)
    # initializes, so they're set as early as possible, before the import below.
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("Xet disabled (HF_HUB_DISABLE_XET=1) -- using plain HTTP transfer.")
    if args.high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        print("Xet high-performance mode enabled (HF_XET_HIGH_PERFORMANCE=1).")

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("huggingface_hub isn't installed. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {args.model_name} ... (Ctrl+C and re-run this same command to resume if it stalls)")
    start = time.time()
    try:
        local_path = snapshot_download(
            repo_id=args.model_name,
            revision=args.revision,
            allow_patterns=[
                "*.json", "*.safetensors", "*.safetensors.index.json",
                "*.model", "*.txt", "tokenizer*", "vocab*", "merges.txt",
            ],
        )
    except HfHubHTTPError as e:
        print(f"\nDownload failed with an HTTP error: {e}", file=sys.stderr)
        print("If this is a 401/403, the model may be gated -- run `hf auth login` first.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume from where it left off.", file=sys.stderr)
        sys.exit(130)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s. Cached at: {local_path}")
    print(
        f"Set `model.local_files_only: true` in your config now if you want training to "
        f"never touch the network again for {args.model_name}."
    )


if __name__ == "__main__":
    main()
