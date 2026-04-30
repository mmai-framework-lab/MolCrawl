# ZINC20 Download and Processing

This module provides a Python-based implementation for downloading and processing ZINC20 chemical data files. It replaces the previous shell script-based approach with a more robust, controllable, and maintainable Python solution.

## Features

- **Pure Python Implementation**: No dependency on external shell scripts
- **Parallel Downloads**: Configurable number of parallel download threads
- **Robust Error Handling**: Automatic retries, timeout handling, and error reporting
- **Progress Monitoring**: Real-time download status and completion tracking
- **File Integrity Checking**: Verification of downloaded files and skip existing files
- **Parquet Conversion**: Efficient conversion to parquet format for data analysis

## Quick Start

### 1. Setup Environment

```bash
# Set the required environment variable
export LEARNING_SOURCE_DIR="learning_source"
```

### 2. Check Download Status

```bash
python molcrawl/data/compounds/dataset/organix13/zinc/download_and_convert_to_parquet.py --status
```

### 3. Download ZINC Files

```bash
# Download with default 1 second delay between downloads
python molcrawl/data/compounds/dataset/organix13/zinc/download_and_convert_to_parquet.py --download

# Download with custom delay to avoid server overload
python molcrawl/data/compounds/dataset/organix13/zinc/download_and_convert_to_parquet.py --download --delay 2.0
```

### 4. Convert to Parquet

```bash
python molcrawl/data/compounds/dataset/organix13/zinc/download_and_convert_to_parquet.py --convert /path/to/output/directory
```

## Module Usage

```python
from molcrawl.data.compounds.dataset.organix13.zinc.download_and_convert_to_parquet import (
    download_zinc_files,
    convert_zinc_to_parquet,
    check_download_status
)

# Check current download status
status = check_download_status()
print(f"Downloaded: {status['downloaded']}/{status['total_expected']} files")

# Download files sequentially with 2 second delay
successful, failed = download_zinc_files(delay_between_downloads=2.0)

# Convert to parquet format
parquet_file = convert_zinc_to_parquet("/path/to/output")
```

## Implementation Details

### File Organization

The ZINC20 dataset consists of approximately 300 files organized as:

- File naming pattern: `XYZW.txt` (4-character combinations)
- Directory structure: Files grouped by first two characters (e.g., `AA/AAAA.txt`)
- Download URLs: `https://files.docking.org/2D/{dir}/{filename}`

### Download Strategy

1. **File List Generation**: Automatically generates the complete list of 300 ZINC files
2. **Sequential Processing**: Downloads files one by one to avoid server overload and 503 errors
3. **Smart Skipping**: Skips existing non-empty files to resume interrupted downloads
4. **Retry Logic**: Implements exponential backoff for failed downloads with longer delays for server errors
5. **Progress Tracking**: Real-time monitoring of download progress
6. **Rate Limiting**: Configurable delay between downloads to respect server limits

### Error Handling

- **Network Timeouts**: 60-second timeout per request with retry logic
- **HTTP Errors**: Proper handling of 404, 500, 503, and other HTTP status codes
- **Server Overload**: Special handling for 503/429 errors with extended retry delays
- **File System Errors**: Graceful handling of disk space and permission issues
- **Partial Downloads**: Detection and re-download of incomplete files

## Configuration

### Environment Variables

- `LEARNING_SOURCE_DIR`: Base directory name for data storage (required)

### Command Line Options

```text
--download          Start downloading ZINC files
--convert PATH      Convert downloaded files to parquet format
--status           Show current download status
--delay SECONDS    Delay in seconds between downloads (default: 1.0)
```

## Migration from Shell Script

The new Python implementation provides several advantages over the previous shell script approach:

1. **Better Control**: Full programmatic control over download process
2. **Error Recovery**: Robust error handling and retry mechanisms
3. **Progress Monitoring**: Real-time status updates and completion tracking
4. **Maintainability**: Easier to modify, extend, and debug
5. **Integration**: Seamless integration with Python data processing pipelines
6. **Server Friendly**: Sequential downloads with configurable delays to avoid 503 errors

### Backward Compatibility

The output file structure remains the same as the shell script version:

- Files are saved to `{COMPOUNDS_DATASET_DIR}/zinc20/`
- Directory structure matches the original layout
- File formats and content are identical

## Dependencies

Required Python packages:

- `requests`: For HTTP downloads
- `dask[dataframe]`: For efficient parquet processing
- `concurrent.futures`: For parallel processing (built-in)

Install dependencies:

```bash
pip install requests dask[dataframe]
```

## Performance

- **Download Speed**: Sequential downloads prevent server overload (503 errors)
- **Memory Usage**: Streaming downloads minimize memory footprint
- **Disk Usage**: ~2-3 GB for complete ZINC20 dataset
- **Processing Time**: Complete download typically takes 2-4 hours with 1-second delays (respectful to server)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `LEARNING_SOURCE_DIR` environment variable is set
2. **Download Failures**: Check network connectivity and retry with `--download`
3. **Disk Space**: Ensure sufficient disk space (~3GB) in target directory
4. **Permission Errors**: Verify write permissions to output directory

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Recovery

The system is designed to be resumable:

- Re-run `--download` to continue interrupted downloads
- Existing files are automatically skipped
- Use `--status` to check completion before restarting

## Testing

Run the test suite to verify functionality:

```bash
# Download status check
python molcrawl/data/compounds/dataset/organix13/zinc/download_and_convert_to_parquet.py --status

# Dry run equivalent: run with a larger delay and monitor logs
python molcrawl/data/compounds/dataset/organix13/zinc/download_and_convert_to_parquet.py --download --delay 2.0
```
