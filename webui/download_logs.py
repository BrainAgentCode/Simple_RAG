#!/usr/bin/env python3
"""
Download logs from Supabase and save them locally.
This script can be used to backup logs or analyze them offline.
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def download_logs(output_dir="downloaded_logs", table=None):
    """Download logs from Supabase and save them locally"""
    
    # Try to get Supabase keys from various sources
    SUPABASE_URL = "https://ylxcsjarxlrdrtmkdfjk.supabase.co"
    supabase_key = None
    
    # 1. Try environment variables (service key for full access)
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    # 2. Fallback to hardcoded key if needed
    if not supabase_key:
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlseGNzamFyeGxyZHJ0bWtkZmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzk2MjUxNCwiZXhwIjoyMDUzNTM4NTE0fQ.6g7xMOwtjd0Wgj6DcVRDo3z1KfzbDSKEhj3z79KcJek"

    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, supabase_key)
        print(f"Connected to Supabase at {SUPABASE_URL}")
        
        # Create output directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of tables to download
        tables = ["user_answers", "user_evaluations", "system_logs", "rag_queries"]
        
        # If table is specified, only download that table
        if table and table in tables:
            tables = [table]
        
        # Download data from each table
        for table_name in tables:
            try:
                print(f"Downloading data from {table_name}...")
                response = supabase.table(table_name).select("*").execute()
                
                if hasattr(response, 'error') and response.error:
                    print(f"Error downloading {table_name}: {response.error}")
                    continue
                
                # Get data and save to file
                data = response.data
                filename = os.path.join(output_dir, f"{table_name}.json")
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Saved {len(data)} records from {table_name} to {filename}")
            
            except Exception as e:
                print(f"❌ Error downloading {table_name}: {e}")
        
        print(f"All logs downloaded to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download logs from Supabase")
    parser.add_argument("--output", "-o", help="Output directory name", default="downloaded_logs")
    parser.add_argument("--table", "-t", help="Specific table to download", choices=["user_answers", "user_evaluations", "system_logs", "rag_queries"])
    args = parser.parse_args()
    
    download_logs(args.output, args.table) 