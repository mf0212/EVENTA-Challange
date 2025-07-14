"""
Data downloader for Event-Enriched Image Captioning Challenge
============================================================

This module provides utilities for downloading and organizing challenge datasets.
"""

import os
import subprocess
import zipfile
import shutil
from typing import Dict, List, Optional
import yaml
from ..utils.logging_utils import get_logger, LoggerMixin

logger = get_logger(__name__)


class DataDownloader(LoggerMixin):
    """
    Data downloader for challenge datasets from Google Drive.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data downloader.
        
        Args:
            config_path: Path to data configuration file
        """
        self.config = self._load_config(config_path)
        self.base_dir = self.config.get('base_dir', 'data')
        self.paths = self.config.get('paths', {})
        self.files = self.config.get('files', {})
        self.gdrive_files = self.config.get('gdrive_files', {})
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'base_dir': 'data',
            'paths': {
                'raw': {'base': 'data/raw'},
                'processed': {'base': 'data/processed'}
            }
        }
    
    def _ensure_gdown_installed(self) -> bool:
        """Ensure gdown is installed."""
        try:
            import gdown
            return True
        except ImportError:
            self.logger.info("Installing gdown...")
            try:
                subprocess.run(['pip', 'install', 'gdown'], check=True)
                return True
            except subprocess.CalledProcessError:
                self.logger.error("Failed to install gdown")
                return False
    
    def download_file(self, file_id: str, destination: str) -> bool:
        """
        Download a file from Google Drive using gdown.
        
        Args:
            file_id: Google Drive file ID
            destination: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_gdown_installed():
            return False
        
        # Create parent directory
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            self.logger.info(f"File {destination} already exists. Skipping download.")
            return True
        
        self.logger.info(f"Downloading to {destination}...")
        
        try:
            import gdown
            gdown.download(id=file_id, output=destination, quiet=False)
            
            # Verify download
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                self.logger.info("Download completed successfully.")
                return True
            else:
                self.logger.error("Download failed or file is empty.")
                if os.path.exists(destination):
                    os.remove(destination)
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: str, extract_dir: str) -> bool:
        """
        Extract a zip file.
        
        Args:
            zip_path: Path to zip file
            extract_dir: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(zip_path):
            self.logger.error(f"Zip file {zip_path} does not exist.")
            return False
        
        os.makedirs(extract_dir, exist_ok=True)
        
        self.logger.info(f"Extracting {zip_path} to {extract_dir}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            self.logger.info("Extraction completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path}: {str(e)}")
            return False
    
    def copy_file(self, source: str, destination: str) -> bool:
        """
        Copy a file to processed directory.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(source):
            self.logger.error(f"Source file {source} does not exist.")
            return False
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        self.logger.info(f"Copying {source} to {destination}...")
        
        try:
            shutil.copy2(source, destination)
            self.logger.info("Copy completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error copying {source} to {destination}: {str(e)}")
            return False
    
    def create_directory_structure(self) -> None:
        """Create the required directory structure."""
        self.logger.info("Creating directory structure...")
        
        directories = [
            "data/raw/train",
            "data/raw/database", 
            "data/raw/query",
            "data/raw/private",
            "data/processed/train",
            "data/processed/database",
            "data/processed/query", 
            "data/processed/private",
            "outputs/submissions",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info("Directory structure created.")
    
    def download_all(self) -> bool:
        """
        Download all challenge datasets.
        
        Returns:
            True if all downloads successful, False otherwise
        """
        self.logger.info("Starting downloads...")
        
        # Create directory structure
        self.create_directory_structure()
        
        success = True
        
        # Download training data
        self.logger.info("Downloading training data...")
        if not self.download_file(
            self.gdrive_files['train']['gt_train'],
            "data/raw/train/gt_train.csv"
        ):
            success = False
        else:
            self.copy_file("data/raw/train/gt_train.csv", "data/processed/train/gt_train.csv")
        
        if not self.download_file(
            self.gdrive_files['train']['images'],
            "data/raw/train/train_compressed_scaled_images.zip"
        ):
            success = False
        else:
            self.extract_zip(
                "data/raw/train/train_compressed_scaled_images.zip",
                "data/processed/train"
            )
        
        # Download database
        self.logger.info("Downloading database...")
        if not self.download_file(
            self.gdrive_files['database']['articles'],
            "data/raw/database/database.json"
        ):
            success = False
        else:
            self.copy_file("data/raw/database/database.json", "data/processed/database/database.json")
        
        if not self.download_file(
            self.gdrive_files['database']['images'],
            "data/raw/database/database_compressed_scaled_images.zip"
        ):
            success = False
        else:
            self.extract_zip(
                "data/raw/database/database_compressed_scaled_images.zip",
                "data/processed/database"
            )
        
        # Download query data
        self.logger.info("Downloading query data...")
        if not self.download_file(
            self.gdrive_files['query']['queries'],
            "data/raw/query/query.csv"
        ):
            success = False
        else:
            self.copy_file("data/raw/query/query.csv", "data/processed/query/query.csv")
        
        if not self.download_file(
            self.gdrive_files['query']['images'],
            "data/raw/query/images.zip"
        ):
            success = False
        else:
            self.extract_zip("data/raw/query/images.zip", "data/processed/query")
        
        # Download private data
        self.logger.info("Downloading private data...")
        if not self.download_file(
            self.gdrive_files['private']['queries'],
            "data/raw/private/query.csv"
        ):
            success = False
        else:
            self.copy_file("data/raw/private/query.csv", "data/processed/private/query.csv")
        
        if not self.download_file(
            self.gdrive_files['private']['images'],
            "data/raw/private/images.zip"
        ):
            success = False
        else:
            self.extract_zip("data/raw/private/images.zip", "data/processed/private")
        
        if success:
            self.logger.info("All downloads completed successfully!")
            self._print_structure_info()
        else:
            self.logger.error("Some downloads failed. Please check the logs.")
        
        return success
    
    def _print_structure_info(self) -> None:
        """Print information about the created data structure."""
        info = """
Files are organized in the following structure:
- data/raw/: Contains the original downloaded files
  - train/gt_train.csv
  - train/train_compressed_scaled_images.zip
  - query/query.csv
  - query/images.zip
  - database/database.json
  - database/database_compressed_scaled_images.zip
  - private/query.csv
  - private/images.zip

- data/processed/: Contains extracted and processed data
  - train/gt_train.csv (copied)
  - train/[extracted images from train_compressed_scaled_images.zip]
  - query/query.csv (copied)
  - query/[extracted images from images.zip]
  - database/database.json (copied)
  - database/[extracted images from database_compressed_scaled_images.zip]
  - private/query.csv (copied)
  - private/[extracted images from images.zip]
        """
        self.logger.info(info)


def main():
    """Main function for standalone execution."""
    downloader = DataDownloader("config/data_config.yaml")
    success = downloader.download_all()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()