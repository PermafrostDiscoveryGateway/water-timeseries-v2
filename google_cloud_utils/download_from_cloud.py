#!/usr/bin/env python3
"""
Google Cloud Storage Download Script

Downloads all contents from a GCS bucket subfolder to a local directory.

Usage:
    python download_from_gcs.py <source_bucket> <gcp_project> <gcs_subfolder_path> <local_path>

Examples:
    # Download all contents from a subfolder
    python download_from_gcs.py my-bucket my-project folder/subfolder/ /local/download/path/

    # Download from bucket root (use empty string or '.' for subfolder path)
    python download_from_gcs.py my-bucket my-project "" /local/download/path/

    # Download specific subfolder
    python download_from_gcs.py my-bucket my-project uploads/images/ ./downloaded_images/
"""

import os
import sys
import argparse
from pathlib import Path
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError, NotFound


def initialize_gcs_client(project_id):
    """Initialize and return Google Cloud Storage client."""
    try:
        client = storage.Client(project=project_id)
        return client
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        print("\nMake sure you have authenticated. Run:")
        print("  gcloud auth application-default login")
        print("Or set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
        sys.exit(1)


def ensure_local_directory(local_path):
    """Create local directory if it doesn't exist."""
    path = Path(local_path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created local directory: {local_path}")
        except Exception as e:
            print(f"✗ Failed to create directory {local_path}: {e}")
            return False
    elif not path.is_dir():
        print(f"✗ Error: {local_path} exists but is not a directory")
        return False
    return True


def download_blob(blob, local_file_path, preserve_structure=True):
    """Download a single blob to local filesystem."""
    try:
        # Create parent directories if needed
        local_path = Path(local_file_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the blob
        blob.download_to_filename(str(local_path))

        # Get file size for feedback
        file_size = blob.size
        size_str = format_file_size(file_size)
        print(f"✓ Downloaded: {blob.name} -> {local_path} ({size_str})")
        return True
    except Exception as e:
        print(f"✗ Failed to download {blob.name}: {e}")
        return False


def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def download_subfolder(bucket, gcs_subfolder, local_path):
    """
    Download all contents from a GCS subfolder to a local directory.

    Args:
        bucket: GCS bucket object
        gcs_subfolder: Path to subfolder in bucket (can be empty string for root)
        local_path: Local directory path to download to
    """
    # Normalize subfolder path
    if gcs_subfolder and not gcs_subfolder.endswith('/'):
        gcs_subfolder += '/'

    print(f"\n📁 Downloading from: gs://{bucket.name}/{gcs_subfolder}")
    print(f"📂 Downloading to: {local_path}")

    # List all blobs with the given prefix
    try:
        blobs = list(bucket.list_blobs(prefix=gcs_subfolder))
    except GoogleAPIError as e:
        print(f"✗ Error listing blobs: {e}")
        return False

    if not blobs:
        print(f"⚠ No files found in gs://{bucket.name}/{gcs_subfolder}")
        return True

    # Filter out the folder itself (if it exists as a zero-byte placeholder)
    blobs = [b for b in blobs if b.name != gcs_subfolder and not b.name.endswith('/')]

    if not blobs:
        print(f"⚠ No files found (only folder placeholders) in gs://{bucket.name}/{gcs_subfolder}")
        return True

    print(f"\n📋 Found {len(blobs)} file(s) to download\n")

    downloaded_count = 0
    failed_count = 0
    total_size = 0

    for blob in blobs:
        # Calculate relative path within the subfolder
        if gcs_subfolder:
            # Remove the subfolder prefix to get relative path
            relative_path = blob.name[len(gcs_subfolder):]
        else:
            relative_path = blob.name

        # Construct local file path
        local_file_path = os.path.join(local_path, relative_path)

        # Download the blob
        if download_blob(blob, local_file_path):
            downloaded_count += 1
            total_size += blob.size
        else:
            failed_count += 1

    # Print summary
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"✓ Successfully downloaded: {downloaded_count} files")
    if failed_count > 0:
        print(f"✗ Failed: {failed_count} files")
    print(f"💾 Total size: {format_file_size(total_size)}")
    print(f"📂 Location: {os.path.abspath(local_path)}")
    print("=" * 50)

    return failed_count == 0


def download_single_file(bucket, gcs_file_path, local_path):
    """
    Download a single file from GCS to a local path.
    Used when the GCS path points to a specific file.
    """
    try:
        # Check if the blob exists
        blob = bucket.blob(gcs_file_path)
        if not blob.exists():
            print(f"✗ Error: File not found in bucket: {gcs_file_path}")
            return False

        # Determine local file path
        local_file_path = local_path
        if os.path.isdir(local_path):
            # If local path is a directory, use the original filename
            filename = os.path.basename(gcs_file_path)
            local_file_path = os.path.join(local_path, filename)

        print(f"\n📄 Downloading single file: {gcs_file_path}")
        print(f"📂 Downloading to: {local_file_path}")

        return download_blob(blob, local_file_path)

    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download a file or all contents from a GCS bucket subfolder to a local directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('source_bucket', help='Name of the GCS bucket')
    parser.add_argument('gcp_project', help='Google Cloud Project ID')
    parser.add_argument('gcs_subfolder_path',
                        help='Path to subfolder in the bucket (use "" or "." for root)')
    parser.add_argument('local_path', help='Local directory path to download to')

    args = parser.parse_args()

    # Ensure local directory exists
    if not ensure_local_directory(args.local_path):
        sys.exit(1)

    # Initialize GCS client
    print(f"🔌 Connecting to Google Cloud project: {args.gcp_project}")
    client = initialize_gcs_client(args.gcp_project)

    # Get the bucket
    try:
        bucket = client.bucket(args.source_bucket)
        # Check if bucket exists
        if not bucket.exists():
            print(f"✗ Error: Bucket '{args.source_bucket}' does not exist or you don't have access")
            sys.exit(1)
        print(f"✓ Connected to bucket: {args.source_bucket}")
    except GoogleAPIError as e:
        print(f"✗ Error accessing bucket: {e}")
        sys.exit(1)

    # Normalize GCS path
    gcs_path = args.gcs_subfolder_path.strip()
    if gcs_path == '.':
        gcs_path = ''

    # Check if the GCS path is a specific file or a folder
    # We'll try to detect by checking if there's an exact blob match
    if gcs_path and not gcs_path.endswith('/'):
        # Check if this exact path exists as a file
        test_blob = bucket.blob(gcs_path)
        if test_blob.exists():
            # It's a file, download single file
            success = download_single_file(bucket, gcs_path, args.local_path)
            sys.exit(0 if success else 1)

    # Otherwise, treat as folder download
    success = download_subfolder(bucket, gcs_path, args.local_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()