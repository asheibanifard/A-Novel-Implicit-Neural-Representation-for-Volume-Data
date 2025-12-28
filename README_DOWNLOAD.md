# Visible Human MRI DICOM Dataset Downloader

Three different methods to download all MRI DICOM slices from the Visible Human Project dataset.

## Dataset Information

**Source:** https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_DICOM/MRI/

**Contents:**
- PD (Proton Density weighted)
- T1 (T1-weighted)
- T2 (T2-weighted)
- T2_512 (T2-weighted at 512×512 resolution)

---

## Method 1: Simple Bash Script (Recommended for most users)

**Best for:** Quick downloads, simplicity, built-in resume capability

### Prerequisites
```bash
# Requires: wget (usually pre-installed on Linux/Mac)
wget --version
```

### Usage
```bash
# Make executable
chmod +x download_visible_human_wget.sh

# Download to default directory (visible_human_mri)
./download_visible_human_wget.sh

# Download to custom directory
./download_visible_human_wget.sh /path/to/output
```

### Features
- Automatically resumes interrupted downloads
- Rate limiting (2 MB/s) to be respectful to server
- Random wait times between requests
- Skips already downloaded files

---

## Method 2: Basic Python Script

**Best for:** Users who prefer Python, simple cross-platform solution

### Prerequisites
```bash
pip install requests beautifulsoup4 tqdm
```

### Usage
```bash
# Basic usage
python download_visible_human_mri.py

# Custom output directory
python download_visible_human_mri.py --output-dir /path/to/output

# Custom URL (e.g., for CT data instead)
python download_visible_human_mri.py --url "https://data.lhncbc.nlm.nih.gov/public/..."
```

### Features
- Progress bars for each file
- Automatic directory creation
- Skip already downloaded files
- Clean console output

---

## Method 3: Advanced Python Script (Recommended for large downloads)

**Best for:** Parallel downloads, resume capability, robust error handling

### Prerequisites
```bash
pip install requests beautifulsoup4 tqdm
```

### Usage
```bash
# Basic usage (4 parallel downloads)
python download_visible_human_advanced.py

# Faster download with 8 parallel workers
python download_visible_human_advanced.py --workers 8

# Custom output directory
python download_visible_human_advanced.py --output-dir /path/to/output

# Fresh download (ignore previous progress)
python download_visible_human_advanced.py --no-resume
```

### Features
- **Parallel downloads** (configurable worker threads)
- **Automatic resume** - safely interrupt and resume later
- **Download manifest** - tracks all downloads in JSON file
- **Retry failed downloads** - just run the script again
- **Progress tracking** - real-time success/failure counts
- **Error logging** - detailed error information

### Resume Downloads
If interrupted, simply run the same command again:
```bash
python download_visible_human_advanced.py
# Will automatically resume from where it left off
```

---

## Comparison

| Feature | Bash Script | Basic Python | Advanced Python |
|---------|-------------|--------------|-----------------|
| Parallel downloads | ✗ | ✗ | ✓ |
| Resume capability | ✓ | ✗ | ✓ |
| Progress tracking | Basic | Good | Excellent |
| Error handling | Basic | Good | Excellent |
| Setup complexity | Minimal | Low | Low |
| Cross-platform | Linux/Mac | All | All |
| Download manifest | ✗ | ✗ | ✓ |

---

## Expected Download Size

The complete MRI dataset contains hundreds of DICOM files:
- Each sequence (PD, T1, T2, T2_512) contains multiple slices
- Typical slice file size: 100-500 KB
- **Estimated total size: 200-500 MB**

Download time depends on:
- Your internet connection
- Server load
- Number of parallel workers (for advanced script)

---

## Directory Structure After Download

```
visible_human_mri/
├── PD/
│   ├── slice_001.dcm
│   ├── slice_002.dcm
│   └── ...
├── T1/
│   ├── slice_001.dcm
│   └── ...
├── T2/
│   └── ...
├── T2_512/
│   └── ...
└── download_manifest.json  (only for advanced script)
```

---

## Working with Downloaded DICOM Files

### Python Example
```python
import pydicom
import numpy as np
from pathlib import Path

# Load a single DICOM file
dcm = pydicom.dcmread('visible_human_mri/T1/slice_001.dcm')
image = dcm.pixel_array

# Load entire sequence
t1_dir = Path('visible_human_mri/T1')
slices = []
for dcm_file in sorted(t1_dir.glob('*.dcm')):
    slices.append(pydicom.dcmread(dcm_file).pixel_array)

volume = np.stack(slices, axis=0)
print(f"Volume shape: {volume.shape}")
```

---

## Troubleshooting

### Download keeps failing
- Try reducing parallel workers: `--workers 2`
- Check internet connection
- Server might be experiencing high load - try again later

### Permission denied
```bash
chmod +x download_visible_human_wget.sh
chmod +x download_visible_human_advanced.py
```

### Python dependencies issues
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install requests beautifulsoup4 tqdm
```

### Disk space issues
- Check available space: `df -h`
- Consider downloading specific sequences only
- Modify scripts to download specific subdirectories

---

## Citation

If you use this dataset in your research, please cite:

```
The Visible Human Project®
U.S. National Library of Medicine
https://www.nlm.nih.gov/research/visible/visible_human.html
```

---

## License

These scripts are provided as-is for downloading publicly available data from the Visible Human Project. Please respect the server by:
- Not running multiple instances simultaneously
- Using reasonable rate limits
- Downloading during off-peak hours when possible

The Visible Human data itself has its own license terms - please review at:
https://www.nlm.nih.gov/databases/download/vhp.html
