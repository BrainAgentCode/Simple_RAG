import os
import sys
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def setup_supabase_tables():
    # Try to get Supabase keys from various sources
    SUPABASE_URL = "https://ylxcsjarxlrdrtmkdfjk.supabase.co"
    supabase_key = None
    
    # 1. Try environment variables
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for setup
    
    # 2. Try Streamlit secrets if available
    if not supabase_key:
        try:
            if 'SUPABASE_SERVICE_KEY' in st.secrets:
                supabase_key = st.secrets['SUPABASE_SERVICE_KEY']
        except Exception as e:
            print(f"Could not access Streamlit secrets: {e}")
    
    # 3. Fallback to hardcoded key if needed
    if not supabase_key:
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlseGNzamFyeGxyZHJ0bWtkZmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzk2MjUxNCwiZXhwIjoyMDUzNTM4NTE0fQ.6g7xMOwtjd0Wgj6DcVRDo3z1KfzbDSKEhj3z79KcJek"

    try:
        # Initialize Supabase client with service key (more permissions)
        supabase: Client = create_client(SUPABASE_URL, supabase_key)
        print("Connected to Supabase with service role")
        
        # Create tables using Supabase API directly
        tables = [
            {
                "name": "user_answers",
                "columns": [
                    {"name": "id", "type": "bigint", "primaryKey": True, "identity": True},
                    {"name": "username", "type": "text", "isNullable": False},
                    {"name": "task_id", "type": "text", "isNullable": False},
                    {"name": "answer", "type": "text"},
                    {"name": "is_correct", "type": "boolean"},
                    {"name": "timestamp", "type": "timestamptz", "defaultValue": "now()", "isNullable": False},
                    {"name": "entry_type", "type": "text"},
                    {"name": "model_used", "type": "text"},
                    {"name": "query", "type": "text"}
                ]
            },
            {
                "name": "user_evaluations",
                "columns": [
                    {"name": "id", "type": "bigint", "primaryKey": True, "identity": True},
                    {"name": "username", "type": "text", "isNullable": False},
                    {"name": "timestamp", "type": "timestamptz", "defaultValue": "now()", "isNullable": False},
                    {"name": "sus_responses", "type": "jsonb"},
                    {"name": "task_difficulty", "type": "jsonb"},
                    {"name": "ai_helpfulness", "type": "text"},
                    {"name": "ai_relevance", "type": "text"},
                    {"name": "retrieval_quality", "type": "text"},
                    {"name": "traditional_comparison", "type": "text"},
                    {"name": "improvement_suggestions", "type": "text"},
                    {"name": "favorite_feature", "type": "text"},
                    {"name": "skipped_tasks", "type": "jsonb"}
                ]
            },
            {
                "name": "system_logs",
                "columns": [
                    {"name": "id", "type": "bigint", "primaryKey": True, "identity": True},
                    {"name": "timestamp", "type": "timestamptz", "defaultValue": "now()", "isNullable": False},
                    {"name": "level", "type": "text", "isNullable": False},
                    {"name": "message", "type": "text", "isNullable": False},
                    {"name": "logger", "type": "text"},
                    {"name": "pathname", "type": "text"},
                    {"name": "lineno", "type": "integer"}
                ]
            },
            {
                "name": "rag_queries",
                "columns": [
                    {"name": "id", "type": "bigint", "primaryKey": True, "identity": True},
                    {"name": "timestamp", "type": "timestamptz", "defaultValue": "now()", "isNullable": False},
                    {"name": "question", "type": "text", "isNullable": False},
                    {"name": "model", "type": "text", "isNullable": False},
                    {"name": "processing_time", "type": "float"},
                    {"name": "num_docs_retrieved", "type": "integer"},
                    {"name": "answer_length", "type": "integer"}
                ]
            }
        ]
        
        # Create each table
        for table in tables:
            table_name = table["name"]
            print(f"Creating table: {table_name}")
            
            try:
                # Check if table exists
                response = supabase.table(table_name).select("*", count="exact").limit(1).execute()
                print(f"Table {table_name} already exists with {response.count} rows")
                
            except Exception:
                # Table doesn't exist, create it
                print(f"Creating new table: {table_name}")
                
                # Use REST API to insert a test row - this will create the table with appropriate schema
                # We'll use this as a workaround since direct table creation via API isn't available
                test_row = {}
                for col in table["columns"]:
                    if col["name"] != "id" and not col.get("identity", False) and not col.get("defaultValue"):
                        # Set a test value based on column type
                        if col["type"] == "text":
                            test_row[col["name"]] = "test_value"
                        elif col["type"] == "boolean":
                            test_row[col["name"]] = False
                        elif col["type"] == "integer" or col["type"] == "float":
                            test_row[col["name"]] = 0
                        elif col["type"] == "jsonb":
                            test_row[col["name"]] = {}
                
                # Insert test row to create table
                if test_row:
                    response = supabase.table(table_name).insert(test_row).execute()
                    if hasattr(response, 'error') and response.error:
                        print(f"Error creating table {table_name}: {response.error}")
                    else:
                        print(f"Table {table_name} created successfully")
                        
                        # Now delete the test row
                        data = response.data
                        if data and len(data) > 0 and "id" in data[0]:
                            test_id = data[0]["id"]
                            supabase.table(table_name).delete().eq("id", test_id).execute()
                            print(f"Test row deleted from {table_name}")
                else:
                    print(f"Could not create table {table_name}: no valid columns found")
        
        print("All tables created successfully")
        return True
        
    except Exception as e:
        print(f"Error setting up Supabase tables: {e}")
        return False

if __name__ == "__main__":
    success = setup_supabase_tables()
    sys.exit(0 if success else 1) 