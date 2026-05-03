# scripts/run_build_metadata.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.build_metadata import main_build_metadata

if __name__ == "__main__":
    main_build_metadata()