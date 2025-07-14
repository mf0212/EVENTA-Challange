#!/bin/bash
# Script to download dataset files from Google Drive using gdown

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    uv add gdown
fi

# Function to download a file from Google Drive using gdown
download_gdrive_file() {
    FILEID=$1
    DESTINATION=$2
    
    # Create parent directory if it doesn't exist
    mkdir -p "$(dirname "$DESTINATION")"
    
    # Check if file already exists
    if [ -f "$DESTINATION" ]; then
        echo "File $DESTINATION already exists. Skipping download."
        return 0
    fi
    
    echo "Downloading to $DESTINATION..."
    gdown --id "$FILEID" --output "$DESTINATION"
    
    # Verify download was successful
    if [ -s "$DESTINATION" ]; then
        echo "Download completed successfully."
    else
        echo "ERROR: Failed to download or file is empty."
        rm -f "$DESTINATION"  # Remove empty file
        return 1
    fi
}

# Function to extract zip files
extract_zip() {
    ZIP_FILE=$1
    EXTRACT_DIR=$2
    
    # Check if zip file exists
    if [ ! -f "$ZIP_FILE" ]; then
        echo "ERROR: Zip file $ZIP_FILE does not exist. Skipping extraction."
        return 1
    fi
    
    # Create extract directory if it doesn't exist
    mkdir -p "$EXTRACT_DIR"
    
    echo "Extracting $ZIP_FILE to $EXTRACT_DIR..."
    unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully."
    else
        echo "ERROR: Failed to extract $ZIP_FILE."
        return 1
    fi
}

# Function to copy file to processed directory
copy_to_processed() {
    SOURCE=$1
    DESTINATION=$2
    
    # Check if source file exists
    if [ ! -f "$SOURCE" ]; then
        echo "ERROR: Source file $SOURCE does not exist. Skipping copy."
        return 1
    fi
    
    # Create parent directory if it doesn't exist
    mkdir -p "$(dirname "$DESTINATION")"
    
    # Copy file
    echo "Copying $SOURCE to $DESTINATION..."
    cp "$SOURCE" "$DESTINATION"
    
    if [ $? -eq 0 ]; then
        echo "Copy completed successfully."
    else
        echo "ERROR: Failed to copy $SOURCE to $DESTINATION."
        return 1
    fi
}

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw/train
mkdir -p data/raw/database
mkdir -p data/raw/query
mkdir -p data/raw/private
mkdir -p data/processed/train
mkdir -p data/processed/database
mkdir -p data/processed/query
mkdir -p data/processed/private
echo "Directory structure created."

# Download files
echo "Starting downloads..."

# gt_train.csv
download_gdrive_file "1lypEb4iLe0mplDl_To7gxCiSL1vTaLnz" "data/raw/train/gt_train.csv"
# Copy gt_train.csv to processed folder
if [ -f "data/raw/train/gt_train.csv" ]; then
    copy_to_processed "data/raw/train/gt_train.csv" "data/processed/train/gt_train.csv"
fi

# train_compressed_scaled_images.zip
download_gdrive_file "1aYIHFp1vPrgrv1p_mhNXMusV0lA64nTM" "data/raw/train/train_compressed_scaled_images.zip"
if [ -f "data/raw/train/train_compressed_scaled_images.zip" ]; then
    extract_zip "data/raw/train/train_compressed_scaled_images.zip" "data/processed/train"
fi

# query.csv
download_gdrive_file "1SjFE2qqwIFKAfhERf8zTgUoB0YrDxB0s" "data/raw/query/query.csv"
# Copy query.csv to processed folder
if [ -f "data/raw/query/query.csv" ]; then
    copy_to_processed "data/raw/query/query.csv" "data/processed/query/query.csv"
fi

# images.zip
download_gdrive_file "1MVLtLhbtIosZL1tWF2S8VaWZKVD1Uson" "data/raw/query/images.zip"
if [ -f "data/raw/query/images.zip" ]; then
    extract_zip "data/raw/query/images.zip" "data/processed/query"
fi

# database.json (large file - 1.0GB)
download_gdrive_file "1GvlG42enj1iwwS3eTIFlLgk04oASCwQE" "data/raw/database/database.json"
# Copy database.json to processed folder
if [ -f "data/raw/database/database.json" ]; then
    copy_to_processed "data/raw/database/database.json" "data/processed/database/database.json"
fi

# database_compressed_scaled_images.zip
download_gdrive_file "1WBlOARXa699KsOsK0LfiJxUXK-JU8XSf" "data/raw/database/database_compressed_scaled_images.zip"
if [ -f "data/raw/database/database_compressed_scaled_images.zip" ]; then
    extract_zip "data/raw/database/database_compressed_scaled_images.zip" "data/processed/database"
fi

# PRIVATE SET DOWNLOADS
echo ""
echo "Downloading private set files..."

# private query.csv
download_gdrive_file "1uc-o4_93c7RKaffxGhPXHC572sBT228j" "data/raw/private/query.csv"
# Copy private query.csv to processed folder
if [ -f "data/raw/private/query.csv" ]; then
    copy_to_processed "data/raw/private/query.csv" "data/processed/private/query.csv"
fi

# private images.zip
download_gdrive_file "1mi2JWvX0shcsIyTi2olELOExG-vCDVus" "data/raw/private/images.zip"
if [ -f "data/raw/private/images.zip" ]; then
    extract_zip "data/raw/private/images.zip" "data/processed/private"
fi

echo ""
echo "All downloads, extractions, and copies completed!"
echo ""
echo "Files are organized in the following structure:"
echo "- data/raw/: Contains the original downloaded files"
echo "  - train/gt_train.csv"
echo "  - train/train_compressed_scaled_images.zip"
echo "  - query/query.csv"
echo "  - query/images.zip"
echo "  - database/database.json"
echo "  - database/database_compressed_scaled_images.zip"
echo "  - private/query.csv"
echo "  - private/images.zip"
echo ""
echo "- data/processed/: Contains extracted and processed data"
echo "  - train/gt_train.csv (copied)"
echo "  - train/[extracted images from train_compressed_scaled_images.zip]"
echo "  - query/query.csv (copied)"
echo "  - query/[extracted images from images.zip]"
echo "  - database/database.json (copied)"
echo "  - database/[extracted images from database_compressed_scaled_images.zip]"
echo "  - private/query.csv (copied)"
echo "  - private/[extracted images from images.zip]"