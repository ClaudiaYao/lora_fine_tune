"""Upload the trained LoRA adapter(s), results and logs to AWS S3.

Bucket and prefix come from .env (S3_BUCKET, S3_PREFIX); credentials from AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY / AWS_REGION in .env, or the normal AWS credential chain if those are blank.

The S3 layout mirrors the local folders, so `aws s3 sync s3://<bucket>/<prefix> ./` restores it:
    s3://<bucket>/<prefix>/outputs/<run>/adapter/...   <- the trained model
    s3://<bucket>/<prefix>/results/<run>.json
    s3://<bucket>/<prefix>/logs/...

Usage:
    python upload_to_s3.py                    # every run in outputs/, plus results/ and logs/
    python upload_to_s3.py --runs unsloth     # one run only
    python upload_to_s3.py --model-only       # skip results/ and logs/
    python upload_to_s3.py --dry-run          # list what would be uploaded
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def make_client():
    load_dotenv(ROOT / ".env")
    kwargs = {"service_name": "s3"}
    if os.getenv("AWS_REGION"):
        kwargs["region_name"] = os.environ["AWS_REGION"]
    # Fall back to the default AWS credential chain if .env values are blank.
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        if os.getenv("AWS_SESSION_TOKEN"):
            kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    return boto3.client(**kwargs)


def collect_files(args):
    """Local files to upload: each run's folder (minus checkpoints), then results/ and logs/."""
    outputs = ROOT / "outputs"
    if args.runs:
        run_dirs = [outputs / r for r in args.runs]
    else:
        run_dirs = sorted(p for p in outputs.glob("*") if p.is_dir())
    if not run_dirs:
        raise SystemExit(f"No trained runs found in {outputs}. Run train-*.py first.")

    files = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"No such run folder: {run_dir}")
        for path in sorted(run_dir.rglob("*")):
            parts = path.relative_to(run_dir).parts
            if "checkpoints" in parts and not args.include_checkpoints:
                continue
            files.append(path)

    if not args.model_only:
        for folder in ("results", "logs"):
            files += sorted((ROOT / folder).rglob("*"))

    return [p for p in files if p.is_file() and not p.name.startswith(".")]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", help="run names under outputs/ (default: all)")
    p.add_argument("--model-only", action="store_true", help="upload outputs/ only, not results/ and logs/")
    p.add_argument("--include-checkpoints", action="store_true", help="also upload outputs/<run>/checkpoints/")
    p.add_argument("--skip-existing", action="store_true", help="skip keys that already exist in the bucket")
    p.add_argument("--bucket", help="overrides S3_BUCKET from .env")
    p.add_argument("--prefix", help="overrides S3_PREFIX from .env")
    p.add_argument("--dry-run", action="store_true", help="print what would be uploaded and exit")
    args = p.parse_args()

    s3 = make_client()  # also loads .env
    bucket = args.bucket or os.getenv("S3_BUCKET")
    if not bucket:
        raise SystemExit("Set S3_BUCKET in .env or pass --bucket")
    prefix = (args.prefix if args.prefix is not None else os.getenv("S3_PREFIX", "")).strip("/")

    files = collect_files(args)
    if not files:
        raise SystemExit("Nothing to upload.")
    keys = {path: "/".join(filter(None, [prefix, path.relative_to(ROOT).as_posix()])) for path in files}
    total = sum(path.stat().st_size for path in files)
    print(f"{len(files)} files, {human(total)} -> s3://{bucket}/{prefix}")

    if args.dry_run:
        for path, key in keys.items():
            print(f"  {human(path.stat().st_size):>9}  {key}")
        return

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        hint = {"403": "no permission (needs s3:ListBucket)", "404": "bucket does not exist",
                "301": "bucket is in a different region; fix AWS_REGION"}.get(code, "")
        raise SystemExit(f"Cannot access bucket '{bucket}': {code} {hint}")

    uploaded = skipped = 0
    for path, key in keys.items():
        if args.skip_existing:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                skipped += 1
                continue
            except ClientError:
                pass  # not there yet
        print(f"  {human(path.stat().st_size):>9}  {key}")
        s3.upload_file(str(path), bucket, key)
        uploaded += 1

    print(f"\nUploaded {uploaded} files" + (f", skipped {skipped} existing" if skipped else "") + ".")
    print(f"Restore with:  aws s3 sync s3://{bucket}/{prefix} ./")


if __name__ == "__main__":
    main()
