#!/usr/bin/env bash
set -euo pipefail

# Set the output file name
output_file="all_files_combined.txt"

# Remove the output file if it already exists
rm -f "$output_file"

# Get all tracked files in the repository, excluding certain files and directories
git ls-tree -r HEAD --name-only \
  | grep -Ev '^(node_modules|dist|sound)/|\.(lock|log|env|DS_Store)$|(\.pt|\.jpg|\.lock)$' \
  | while IFS= read -r filename; do
    # Check if the file still exists (it might have been deleted)
    if [ -f "$filename" ]; then
        # Append the filename as a header
        echo "### $filename" >> "$output_file"

        # Append the file contents
        cat "$filename" >> "$output_file"

        # Add two newlines for separation
        echo "" >> "$output_file"
        echo "" >> "$output_file"
    fi
done

echo "All files have been combined into $output_file"

