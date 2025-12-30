#!/bin/bash

# Download all MRI DICOM slices from Visible Human Dataset
# Usage: ./download_visible_human_wget.sh [output_directory]

OUTPUT_DIR="${1:-visible_human_mri}"
BASE_URL="https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_DICOM/MRI/index.html"

echo "=================================================="
echo "Visible Human MRI DICOM Downloader (wget)"
echo "=================================================="
echo "Base URL: $BASE_URL"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This will download all MRI sequences:"
echo "  - PD (Proton Density)"
echo "  - T1 (T1-weighted)"
echo "  - T2 (T2-weighted)"
echo "  - T2_512 (T2-weighted 512x512)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Download cancelled."
    exit 1
fi

echo ""
echo "Starting download..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download recursively
# -r: recursive
# -np: no parent (don't ascend to parent directory)
# -nH: no host directories
# --cut-dirs=5: remove first 5 directory components from path
# -R: reject index.html files
# --wait=1: wait 1 second between downloads (be nice to server)
# -P: prefix (output directory)
# --continue: resume partially downloaded files
# --timestamping: only download newer files (skip already completed)
# --user-agent: set browser-like user agent to avoid 403 errors

wget \
    --recursive \
    --no-parent \
    --no-host-directories \
    --cut-dirs=5 \
    --reject="index.html*" \
    --wait=1 \
    --random-wait \
    --limit-rate=2m \
    --continue \
    --timestamping \
    --directory-prefix="$OUTPUT_DIR" \
    --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
    --execute robots=off \
    "$BASE_URL"

echo ""
echo "=================================================="
echo "Download complete!"
echo "Files saved to: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
find "$OUTPUT_DIR" -type d
echo ""
echo "Total files downloaded:"
find "$OUTPUT_DIR" -type f | wc -l
echo "=================================================="
