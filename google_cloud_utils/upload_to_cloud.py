#!/usr/bin/env python3
"""
Google Cloud Storage Upload Script

Usage:
    python upload_to_gcs.py <source_bucket> <gcp_project> <local_path> <gcs_destination_path>

Examples:
    # Upload a single file
    python upload_to_gcs.py my-bucket my-project /local/file.txt folder/subfolder/file.txt

    # Upload a directory (all contents recursively)
    python upload_to_gcs.py my-bucket my-project /local/directory/ folder/subfolder/
"""

import os
import sys
import argparse
from pathlib import Path
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError


def initialize_gcs_client(project_id):
    """Initialize and return Google Cloud Storage client."""
    try:
        # The client will use default credentials from environment
        # Set GOOGLE_APPLICATION_CREDENTIALS env var or run: gcloud auth application-default login
        client = storage.Client(project=project_id)
        return client
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        print("\nMake sure you have authenticated. Run:")
        print("  gcloud auth application-default login")
        print("Or set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
        sys.exit(1)


def upload_file(bucket, source_path, destination_blob_name):
    """Upload a single file to GCS."""
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_path)
        print(f"✓ Uploaded file: {source_path} -> gs://{bucket.name}/{destination_blob_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload {source_path}: {e}")
        return False


def upload_directory(bucket, source_dir, destination_prefix):
    """Upload all contents of a directory recursively to GCS."""
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Directory does not exist: {source_dir}")
        return False

    if not source_path.is_dir():
        print(f"Error: Not a directory: {source_dir}")
        return False

    uploaded_count = 0
    failed_count = 0

    # Walk through all files in the directory
    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)

            # Calculate relative path to preserve directory structure
            relative_path = os.path.relpath(local_file_path, source_dir)

            # Construct destination path
            if destination_prefix and not destination_prefix.endswith('/'):
                destination_prefix += '/'
            destination_blob = os.path.join(destination_prefix, relative_path).replace('\\', '/')

            # Upload the file
            if upload_file(bucket, local_file_path, destination_blob):
                uploaded_count += 1
            else:
                failed_count += 1

    print(f"\n📊 Upload summary: {uploaded_count} files uploaded, {failed_count} failed")
    return failed_count == 0


def main():
    parser = argparse.ArgumentParser(
        description='Upload a file or directory to Google Cloud Storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('source_bucket', help='Name of the GCS bucket')
    parser.add_argument('gcp_project', help='Google Cloud Project ID')
    parser.add_argument('local_path', help='Local path to a file or directory')
    parser.add_argument('gcs_destination_path', help='Destination path in the bucket (can include folders)')

    args = parser.parse_args()

    # Validate local path exists
    local_path = Path(args.local_path)
    if not local_path.exists():
        print(f"Error: Local path does not exist: {args.local_path}")
        sys.exit(1)

    # Initialize GCS client
    print(f"Connecting to Google Cloud project: {args.gcp_project}")
    client = initialize_gcs_client(args.gcp_project)

    # Get the bucket
    try:
        bucket = client.bucket(args.source_bucket)
        # Check if bucket exists
        if not bucket.exists():
            print(f"Error: Bucket '{args.source_bucket}' does not exist or you don't have access")
            sys.exit(1)
        print(f"✓ Connected to bucket: {args.source_bucket}")
    except GoogleAPIError as e:
        print(f"Error accessing bucket: {e}")
        sys.exit(1)

    # Determine if local path is a file or directory
    if local_path.is_file():
        print(f"Uploading single file: {args.local_path}")
        success = upload_file(bucket, str(local_path), args.gcs_destination_path)
    elif local_path.is_dir():
        print(f"Uploading directory contents: {args.local_path}")
        print(f"Destination prefix: {args.gcs_destination_path}")
        success = upload_directory(bucket, str(local_path), args.gcs_destination_path)
    else:
        print(f"Error: Local path is neither a file nor directory: {args.local_path}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()