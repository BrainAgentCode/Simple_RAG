import streamlit as st
import pandas as pd
from rag_models import get_rag_model
import datetime
import os
import json
import random
from pathlib import Path
from streamlit.components.v1 import html
from dotenv import load_dotenv
import requests
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def load_models_from_provider(provider, config):
    """从提供商加载模型列表"""
    models = []
    error = ""
    
    try:
        if provider == "OpenAI":
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", "https://api.openai.com/v1")
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY', '')
            if not api_key:
                return [], "请先设置OpenAI API Key"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                models.sort()
            else:
                error = f"API返回错误: {response.status_code}"
        
        elif provider == "Anthropic":
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", "https://api.anthropic.com")
            if not api_key:
                api_key = os.getenv('ANTHROPIC_API_KEY', '')
            if not api_key:
                return [], "请先设置Anthropic API Key"
            
            # 尝试调用API获取模型列表
            try:
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    models.sort()
                else:
                    # 如果API调用失败，返回预设模型
                    models = [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307"
                    ]
            except:
                # 出错时返回预设模型
                models = [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
        
        elif provider == "本地模型":
            base_url = config.get("base_url", "http://localhost:8000/v1")
            api_key = config.get("api_key", "not-needed")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                models.sort()
            else:
                error = f"本地模型服务未响应: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        error = "无法连接到模型服务"
    except requests.exceptions.Timeout:
        error = "连接超时"
    except Exception as e:
        error = f"加载失败: {str(e)}"
    
    return models, error

def get_current_model_name():
    """获取当前选中的模型名称"""
    if st.session_state.use_custom_model:
        return st.session_state.custom_model_name
    elif st.session_state.loaded_models:
        return st.session_state.selected_model_name
    else:
        return st.session_state.custom_model_name or "未配置模型"

# Initialize Supabase client
SUPABASE_URL = "https://ylxcsjarxlrdrtmkdfjk.supabase.co"

# Try to get Supabase keys from various sources
supabase_key = None

# 1. Try environment variables
supabase_key = os.getenv('SUPABASE_ANON_KEY')

# 2. Try Streamlit secrets if available
if not supabase_key:
    try:
        if 'SUPABASE_ANON_KEY' in st.secrets:
            supabase_key = st.secrets['SUPABASE_ANON_KEY']
    except Exception as e:
        print(f"Could not access Streamlit secrets: {e}")

# 3. Fallback to hardcoded key if needed
if not supabase_key:
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlseGNzamFyeGxyZHJ0bWtkZmprIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc5NjI1MTQsImV4cCI6MjA1MzUzODUxNH0.N0SLqiMO6KxAlf_hyNTu1W1RZ8MfltuXwtdc1o-7eAs"

supabase: Client = None
try:
    supabase = create_client(SUPABASE_URL, supabase_key)
    st.session_state.supabase_connected = True
    print("Supabase connection successful!")
except Exception as e:
    st.session_state.supabase_connected = False
    print(f"Failed to connect to Supabase: {e}")

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "model_config.json"

def load_config_from_file():
    """从本地文件加载配置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config_to_file():
    """保存当前配置到本地文件"""
    config = {
        "selected_provider": st.session_state.selected_provider,
        "provider_config": st.session_state.provider_config,
        "selected_model_name": st.session_state.selected_model_name,
        "custom_model_name": st.session_state.custom_model_name,
        "use_custom_model": st.session_state.use_custom_model
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置失败: {e}")

# 从文件加载已保存的配置
saved_config = load_config_from_file()

# Initialize session state variables if they don't exist
if 'user_msgs' not in st.session_state:
    st.session_state.user_msgs = []
if 'bot_msgs' not in st.session_state:
    st.session_state.bot_msgs = [f"欢迎使用航空航天知识问答系统。本系统基于检索增强生成（RAG）技术，结合AI与NASA可信文档信息，提供更准确、更可靠的回答。\n\n作为评估的一部分，请在侧边栏完成5个工程任务。如果某个任务特别困难，尝试4次后可以跳过。完成或跳过所有任务后，将出现反馈表单。感谢您的参与！"]
if 'show_song_selection' not in st.session_state:
    st.session_state.show_song_selection = False
if 'song_options' not in st.session_state:
    st.session_state.song_options = []
if "recommended" not in st.session_state:
    st.session_state.recommended = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "openai"

# 模型提供商和加载状态（优先使用已保存的配置）
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = saved_config.get("selected_provider", "OpenAI")
if 'provider_config' not in st.session_state:
    default_config = {
        "OpenAI": {"api_key": "", "base_url": "https://api.openai.com/v1"},
        "Anthropic": {"api_key": "", "base_url": "https://api.anthropic.com"},
        "本地模型": {"base_url": "http://localhost:8000/v1", "api_key": "not-needed"}
    }
    saved_provider_config = saved_config.get("provider_config", {})
    # 合并默认配置和已保存配置
    for provider in default_config:
        if provider in saved_provider_config:
            default_config[provider].update(saved_provider_config[provider])
    st.session_state.provider_config = default_config
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = []
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = saved_config.get("selected_model_name", "")
if 'custom_model_name' not in st.session_state:
    st.session_state.custom_model_name = saved_config.get("custom_model_name", "")
if 'use_custom_model' not in st.session_state:
    st.session_state.use_custom_model = saved_config.get("use_custom_model", False)
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'model_load_error' not in st.session_state:
    st.session_state.model_load_error = ""
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'completed_tasks' not in st.session_state:
    st.session_state.completed_tasks = set()
# Initialize task model assignments if not exists
if 'task_models' not in st.session_state:
    # Randomly assign models to tasks
    st.session_state.task_models = {
        f"task{i}": random.choice(["openai", "llama"]) for i in range(1, 6)
    }
if 'can_select_model' not in st.session_state:
    st.session_state.can_select_model = False
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'show_feedback_popup' not in st.session_state:
    st.session_state.show_feedback_popup = False
if 'skipped_tasks' not in st.session_state:
    st.session_state.skipped_tasks = set()
# Add a flag for switching to About tab after login
if 'switch_to_about' not in st.session_state:
    st.session_state.switch_to_about = False
# Flag to track if we've shown the cloud environment warning
if 'cloud_warning_shown' not in st.session_state:
    st.session_state.cloud_warning_shown = False

# Initialize the RAG model
rag_model = get_rag_model()

# Define hardcoded credentials
USERS = {
    "user1": "password1",
    "user2": "password2",
    "admin": "adminpass",
    "bruker1": "passord1",
    "bruker2": "passord2",
    "bruker3": "passord3",
    "bruker4": "passord4", # Sent to Cephas
    "Snorre": "Snorre123", # Sent
    "Martin": "Martin123", # Sent
    "Christoffer": "Christoffer123", # Sent
    "Marius": "Marius123", # Sent
    "Edvard": "Edvard123", # Sent
    "Daniel": "Daniel123", # Sent
    "Stine": "Stine123", # Sent
    "Eirik": "Eirik123", # Sent
    "Fredrik": "Fredrik123", # Sent
    "Emerson": "Emerson123", # Sent
    "Johan": "Johan123", # Sent
    "Dominykas": "Dominykas123", # Sent
    "Chiran": "Chiran123", # Sent
    "Filip": "Filip123", # Sent
    "Sina": "Sina123", # Sent
    "Håvard": "Håvard123", # Sent
    "Kevin": "Kevin123", # Sent
    "Jonathan": "Jonathan123", # Sent
    "Tord": "Tord123", # Sent
    "Patrik": "Patrik123", # Sent 
    "Kien": "Kien123", # Sent
}

# Define key terms/concepts that should be in correct answers - move this outside the function
key_concepts = {
    "task1": {
        "required": ["PSL", "LF11", "ice particle", "rollback", "wet bulb temperature", "LPC"],
        "min_required": 3,
        "feedback_correct": "You've correctly identified the key factors in engine rollback events.",
        "concept_hints": {
            "PSL": "Consider including information about the Propulsion Systems Laboratory (PSL) test facility data.",
            "LF11": "The LF11 engine model (a specific turbofan engine type) is relevant to this analysis.",
            "ice particle": "What size and type of ice particles were critical in the tests?",
            "rollback": "Describe the conditions that lead to engine rollback events (when engine thrust decreases unexpectedly).",
            "wet bulb temperature": "Temperature conditions, particularly wet bulb temperature (a measure that accounts for humidity), play a key role.",
            "LPC": "The Low Pressure Compressor (LPC) region is important to mention in your analysis."
        }
    },
    "task2": {
        "required": ["CALIPSO", "thermal", "SHM", "DAQ", "standby mode", "safe mode", "heater"],
        "min_required": 3,
        "feedback_correct": "Great analysis of the CALIPSO thermal system performance!",
        "concept_hints": {
            "CALIPSO": "Your answer should specifically address the Cloud-Aerosol Lidar and Infrared Pathfinder Satellite Observation (CALIPSO) payload.",
            "thermal": "Focus on the thermal aspects (temperature control and heat management) of the system performance.",
            "SHM": "Include information about the System Health Monitoring (SHM) mode and its thermal characteristics.",
            "DAQ": "The Data Acquisition (DAQ) mode has specific thermal characteristics worth discussing.",
            "standby mode": "How does the thermal system perform in standby mode (reduced power operation)?",
            "safe mode": "Consider the thermal conditions during safe mode operations (emergency power conservation).",
            "heater": "Heater performance and temperature regulation is a critical aspect to evaluate."
        }
    },
    "task3": {
        "required": ["noise", "engine power", "approach", "takeoff", "broadband", "flight velocity", "airframe"],
        "min_required": 3,
        "feedback_correct": "Excellent understanding of aircraft noise profiles!",
        "concept_hints": {
            "noise": "Be more specific about the types of noise components and sources in aircraft operation.",
            "engine power": "How do different engine power settings affect the overall noise generation?",
            "approach": "Consider noise characteristics during approach conditions (when aircraft is descending to land).",
            "takeoff": "Takeoff conditions (maximum power) have distinct noise profiles worth mentioning.",
            "broadband": "Broadband noise (noise distributed across many frequencies) components vary with different conditions.",
            "flight velocity": "How does the speed of the aircraft affect the noise profile?",
            "airframe": "Don't forget to consider airframe noise contributions (noise from the aircraft body, not engines)."
        }
    },
    "task4": {
        "required": ["UPS", "power quality", "safety", "personnel", "equipment", "mission", "critical"],
        "min_required": 3,
        "feedback_correct": "Your UPS system recommendations are well-justified!",
        "concept_hints": {
            "UPS": "Be specific about Uninterruptible Power Supply (UPS) capabilities and implementation.",
            "power quality": "How does a UPS system improve power quality (voltage stability, frequency regulation)?",
            "safety": "Consider safety implications for mission operations when power fluctuations occur.",
            "personnel": "How does UPS implementation affect personnel safety during power events?",
            "equipment": "Discuss protection of sensitive equipment from power surges or outages.",
            "mission": "Relate your answer to mission requirements and continuity of operations.",
            "critical": "Identify which systems are most critical for UPS protection in a space mission context."
        }
    },
    "task5": {
        "required": ["circuit analysis", "electro-mechanical", "safety", "design review", "manufacturing", "critical"],
        "min_required": 3,
        "feedback_correct": "Your analysis technique recommendations are thorough and appropriate!",
        "concept_hints": {
            "circuit analysis": "Specify which circuit analysis techniques (such as FMEA, fault tree analysis) are most effective.",
            "electro-mechanical": "Address the electro-mechanical aspects (where electrical and mechanical systems interact) of the system.",
            "safety": "How do these analytical techniques improve system safety and reliability?",
            "design review": "When in the design review process should these analyses be applied for maximum benefit?",
            "manufacturing": "Consider techniques that identify potential issues before manufacturing begins.",
            "critical": "Explain why these techniques are especially important for high-reliability, mission-critical systems."
        }
    }
}

def evaluate_task_answer(task_id, answer):
    """
    Evaluate user's answer for a specific task.
    Returns (is_correct, feedback_message, missing_concepts)
    """
    # Check which concepts are present and which are missing
    task_concepts = key_concepts[task_id]
    found_concepts = []
    missing_concepts = []
    
    for concept in task_concepts["required"]:
        if concept.lower() in answer.lower():
            found_concepts.append(concept)
        else:
            missing_concepts.append(concept)
    
    is_correct = len(found_concepts) >= task_concepts["min_required"]
    
    if is_correct:
        return True, task_concepts["feedback_correct"], []
    else:
        # Return the list of missing concepts for targeted feedback
        return False, task_concepts["feedback_correct"], missing_concepts

# Function to log user answers with model information
def log_user_answer(username, task_id, answer, is_correct, model_used=None, query=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "username": username,
        "task_id": task_id,
        "answer": answer,
        "is_correct": is_correct,
        "timestamp": timestamp,
        "entry_type": "query" if task_id.startswith("query") or is_correct == "not_submitted" else "submission"
    }
    
    # Add model and query information if provided
    if model_used:
        log_entry["model_used"] = model_used
    if query:
        log_entry["query"] = query
    
    # Try to log to Supabase first
    if st.session_state.get('supabase_connected', False) and supabase:
        try:
            # Check if table exists by trying to select a single row first
            try:
                supabase.table("user_answers").select("*").limit(1).execute()
                table_exists = True
            except Exception:
                print("user_answers table doesn't exist in Supabase. Falling back to local logging.")
                table_exists = False
            
            if table_exists:
                response = supabase.table("user_answers").insert(log_entry).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"Supabase error: {response.error}")
                    # Fall back to local logging on error
                    _log_to_local_file(username, log_entry, "user_answers")
                else:
                    print(f"Successfully logged to Supabase: {task_id}")
                    return
            else:
                # Fall back to local logging if table doesn't exist
                _log_to_local_file(username, log_entry, "user_answers")
        except Exception as e:
            print(f"Error logging to Supabase: {e}")
            # Fall back to local logging on exception
            _log_to_local_file(username, log_entry, "user_answers")
    else:
        # Fall back to local logging if Supabase is not connected
        _log_to_local_file(username, log_entry, "user_answers")

# Helper function for local file logging
def _log_to_local_file(username, log_entry, log_type):
    log_dir = log_type  # "user_answers" or "user_evaluations"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{username}_{log_type}.json")
    
    # Load existing logs if file exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add new entry and save
    logs.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

# Add this function after your other logging functions
def log_user_evaluation(username, evaluation_data):
    """Log user's final evaluation feedback"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_data["timestamp"] = timestamp
    evaluation_data["username"] = username
    
    # Try to log to Supabase first
    if st.session_state.get('supabase_connected', False) and supabase:
        try:
            # Check if table exists by trying to select a single row first
            try:
                supabase.table("user_evaluations").select("*").limit(1).execute()
                table_exists = True
            except Exception:
                print("user_evaluations table doesn't exist in Supabase. Falling back to local logging.")
                table_exists = False
            
            if table_exists:
                response = supabase.table("user_evaluations").insert(evaluation_data).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"Supabase error: {response.error}")
                    # Fall back to local logging on error
                    _log_to_local_file(username, evaluation_data, "user_evaluations")
                else:
                    print(f"Successfully logged evaluation to Supabase for user: {username}")
                    return True
            else:
                # Fall back to local logging if table doesn't exist
                _log_to_local_file(username, evaluation_data, "user_evaluations")
        except Exception as e:
            print(f"Error logging evaluation to Supabase: {e}")
            # Fall back to local logging on exception
            _log_to_local_file(username, evaluation_data, "user_evaluations")
    else:
        # Fall back to local logging if Supabase is not connected
        _log_to_local_file(username, evaluation_data, "user_evaluations")
    
    return True

def check_all_tasks_completed_or_skipped():
    """Check if all tasks are either completed or skipped"""
    for i in range(1, 6):
        task_key = f"task{i}"
        if (not st.session_state.task_completion[task_key]["correct"] and 
            task_key not in st.session_state.skipped_tasks):
            return False
    return True

def process_rag_query(query, task_id=None):
    """Process a query using the RAG model with the selected model"""
    # 获取当前提供商和模型名称
    provider = st.session_state.selected_provider
    model_name = get_current_model_name()
    config = st.session_state.provider_config.get(provider, {})
    task_to_use = task_id if task_id else st.session_state.current_task_id
    
    result = rag_model.query(query, provider, model_name, config)
    
    # Log the query and model used
    if st.session_state.logged_in:
        log_user_answer(
            st.session_state.username, 
            "query" if task_to_use is None else f"query_{task_to_use}", 
            result, 
            "not_submitted",  # Mark as not submitted since this is just a query
            model_used=f"{provider}/{model_name}",
            query=query
        )
    
    return result

# Function to switch between tabs using JavaScript
def switch_tab(tab_index):
    """Generate JavaScript to switch to the specified tab index"""
    return f"""
    <script>
    var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
    var tabs = tabGroup.getElementsByTagName("button");
    tabs[{tab_index}].click();
    </script>
    """

# Function to stay on the RAG Query tab (index 0) after form submission
def stay_on_rag_tab():
    """Switch to the RAG Query tab after a form submission"""
    # We use index 0 because RAG Query is the first tab (index 0)
    st.markdown(switch_tab(0), unsafe_allow_html=True)

# Function to switch to the About tab (index 1) after login
def go_to_about_tab():
    """Switch to the About tab after successful login"""
    # We use index 1 because About is the second tab (index 1)
    st.markdown(switch_tab(1), unsafe_allow_html=True)

# Login form
if not st.session_state.logged_in:
    st.title("NASA经验教训系统登录")
    
    # Add the message about user evaluation
    st.info("请使用用户名: admin 密码: adminpass 登录。")
    
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submit_button = st.form_submit_button("登录")
        
        if submit_button:
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.switch_to_about = True  # Set flag to switch tabs after rerun
                st.success(f"欢迎, {username}!")
                st.rerun()
            else:
                st.error("用户名或密码错误")
else:
    # Add logout button in the sidebar
    if st.sidebar.button("退出登录"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    # Display current user
    st.sidebar.write(f"当前用户: **{st.session_state.username}**")
    
    # Show cloud environment warning if needed
    is_ollama_available = False
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        is_ollama_available = response.status_code == 200
    except:
        is_ollama_available = False
    
    # Define tabs here, inside the else block
    tab1, tab3 = st.tabs(["RAG查询", "关于"])
    
    # Check if we need to switch to About tab (after successful login)
    if st.session_state.switch_to_about:
        st.session_state.switch_to_about = False  # Reset the flag
        html(switch_tab(1), height=0)  # Switch to About tab (index 1)

    with tab3:
        messages = st.container(height=320)
        messages.chat_message("assistant").write(st.session_state.bot_msgs[0])

        # Add back the columns for better centering
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Custom CSS for a colorful, rainbow gradient button - using more specific selectors
            st.markdown("""
            <style>
            /* Override the default Streamlit button styling completely */
            div[data-testid="element-container"] button[kind="primary"] {
                background: linear-gradient(124deg, 
                    #ff2400, #e81d1d, #e8b71d, #e3e81d, #1de840, 
                    #1ddde8, #2b1de8, #dd00f3, #dd00f3) !important;
                background-size: 1800% 1800% !important;
                animation: rainbow 10s ease infinite !important;
                color: white !important;
                padding: 15px 32px !important;
                text-align: center !important;
                display: inline-block !important;
                font-size: 42px !important;
                font-weight: bold !important;
                margin: 10px 0px !important;
                cursor: pointer !important;
                border-radius: 12px !important;
                border: none !important;
                width: 100% !important;
                height: auto !important;
                transition: all 0.3s !important;
                box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
            }
            
            /* Hover and active states */
            div[data-testid="element-container"] button[kind="primary"]:hover {
                transform: translateY(-5px) !important;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
            }
            div[data-testid="element-container"] button[kind="primary"]:active {
                transform: translateY(3px) !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            }
            
            /* More specific override for Streamlit's button styling */
            div[data-testid="stButton"] > button[kind="primary"] {
                background-image: linear-gradient(124deg, 
                    #ff2400, #e81d1d, #e8b71d, #e3e81d, #1de840, 
                    #1ddde8, #2b1de8, #dd00f3, #dd00f3) !important;
                background-size: 1800% 1800% !important;
                animation: rainbow 10s ease infinite !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Use Streamlit's button with our CSS applied for reliable tab switching
            if st.button("开始", key="rainbow_button", type="primary", use_container_width=True):
                html(switch_tab(0), height=0)

        # Only show user-bot message pairs if there are any
        if st.session_state.user_msgs:
            for i, (user_msg, bot_msg) in enumerate(zip(st.session_state.user_msgs, st.session_state.bot_msgs[1:])):
                messages.chat_message("user").write(user_msg)
                messages.chat_message("assistant").write(bot_msg)

        # Button container below chat
        button_container = st.container(border=True)

        with st.sidebar:
            st.header("用户评估任务", divider="gray")
            
            # Initialize task completion state if not exists
            if 'task_completion' not in st.session_state:
                st.session_state.task_completion = {
                    "task1": {"completed": False, "correct": False, "attempts": 0},
                    "task2": {"completed": False, "correct": False, "attempts": 0},
                    "task3": {"completed": False, "correct": False, "attempts": 0},
                    "task4": {"completed": False, "correct": False, "attempts": 0},
                    "task5": {"completed": False, "correct": False, "attempts": 0}
                }
        
            
            # Task selection and submission system
            selected_task = st.selectbox(
                "选择要完成的任务:",
                ["任务1: 发动机回退调查", 
                 "任务2: 卫星热系统评估",
                 "任务3: 飞机噪声特征评估",
                 "任务4: 关键电源系统设计",
                 "任务5: 电子系统安全审查"]
            )
            
            task_id = f"task{selected_task[5:6]}"  # Extract task number
            # Store the currently selected task in session state
            st.session_state.current_task_id = task_id
            
            # Display task description based on selection
            if selected_task == "任务1: 发动机回退调查":
                st.markdown("""
                ### 任务1: 发动机回退调查
                """)
                # Make the task description non-copyable using HTML/CSS
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                您是一名航空工程师，正在分析结冰条件下的发动机性能。您的团队需要了解哪些颗粒特性会导致发动机回退事件。研究推进系统实验室(PSL)的测试数据，确定导致这些事件的关键冰晶尺寸和温度条件。
                
                预期发现应包括：
                - 对LF11发动机型号推进系统实验室(PSL)数据点的分析
                - 发动机回退的关键颗粒尺寸要求（当推力意外减小时）
                - 低压压缩机(LPC)区域相关的湿球温度范围
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "任务2: 卫星热系统评估":
                st.markdown("""
                ### 任务2: 卫星热系统评估
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                作为热控工程师，您需要评估云-气溶胶激光雷达和红外路径卫星观测(CALIPSO)载荷热系统在不同工作模式下的表现。研究热性能数据，编写关于系统稳定性和余量条件的简要报告。
                
                预期发现应包括：
                - 系统健康监测(SHM)和数据采集(DAQ)模式下的热边界条件性能
                - 各种待机和安全模式下的系统行为（低功率和应急操作）
                - 加热器性能和温度控制有效性
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "任务3: 飞机噪声特征评估":
                st.markdown("""
                ### 任务3: 飞机噪声特征评估
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                您正在为新飞机设计进行降噪工作。您的任务是了解发动机功率设置如何影响不同的噪声成分。研究进近和起飞条件下的噪声特征差异，为您的设计建议提供依据。
                
                预期发现应包括：
                - 低功率设置下进气道宽频成分的行为（分布在多个频率上的噪声）
                - 飞行速度与机体噪声（来自飞机机身的噪声）之间的关系
                - 高起飞功率下的对比噪声水平
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "任务4: 关键电源系统设计":
                st.markdown("""
                ### 任务4: 关键电源系统设计
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                您正在为带有敏感设备的新航天任务设计电源系统。您的项目经理希望得到关于实施不间断电源(UPS)系统的建议。研究UPS在NASA任务中的优势和应用，为您的提案提供论证。
                
                预期发现应包括：
                - 对人员和设备的安全优势
                - 应急操作的关键应用
                - 电能质量改善能力（电压稳定性、频率调节）
                </div>
                """, unsafe_allow_html=True)
            elif selected_task == "任务5: 电子系统安全审查":
                st.markdown("""
                ### 任务5: 电子系统安全审查
                """)
                # Make the task description non-copyable
                st.markdown("""
                <div style="user-select: none; -webkit-user-select: none; -ms-user-select: none;">
                作为系统安全工程师，您正在为关键设计评审做准备，需要为复杂的机电系统推荐适当的分析技术。研究可在制造前识别潜在隐藏电路问题的分析方法。
                
                预期发现应包括：
                - 适用于专门电路分析（如FMEA或故障树分析）的系统类型
                - 项目生命周期中的最佳实施时机
                - 对高关键性系统（失效会导致灾难性后果的系统）的优势
                </div>
                """, unsafe_allow_html=True)
            
            # Task answer submission
            st.write("提交你的发现:")
            user_answer = st.text_area("你的答案", height=150, key=f"answer_{task_id}")
            
            # Create columns for Submit and Skip buttons
            col1, col2 = st.columns(2)
            
            with col1:
                submit_button = st.button("提交答案", key=f"submit_{task_id}")
            
            with col2:
                # Only show skip button if there have been at least 4 unsuccessful attempts
                can_skip = task_id in st.session_state.task_completion and st.session_state.task_completion[task_id]["attempts"] >= 4
                
                if can_skip:
                    skip_button = st.button("跳过任务", key=f"skip_{task_id}")
                else:
                    # Show disabled skip button with attempts counter
                    attempts = 0
                    if task_id in st.session_state.task_completion:
                        attempts = st.session_state.task_completion[task_id]["attempts"]
                    
                    remaining = max(0, 4 - attempts)
                    st.button(
                        f"跳过（还需{remaining}次尝试）",
                        key=f"skip_disabled_{task_id}",
                        disabled=True
                    )
                    skip_button = False
            
            if submit_button:
                # Evaluate the answer
                is_correct, feedback_message, missing_concepts = evaluate_task_answer(task_id, user_answer)
                
                # Update task completion state
                if task_id not in st.session_state.task_completion:
                    st.session_state.task_completion[task_id] = {"completed": False, "correct": False, "attempts": 0}
                
                st.session_state.task_completion[task_id]["attempts"] += 1
                current_attempts = st.session_state.task_completion[task_id]["attempts"]
                st.session_state.task_completion[task_id]["completed"] = True
                st.session_state.task_completion[task_id]["correct"] = is_correct
                
                # Log the user's answer with the model used for this task
                log_user_answer(
                    st.session_state.username, 
                    task_id, 
                    user_answer, 
                    is_correct,
                    model_used=st.session_state.task_models[task_id],
                    query=f"Task submission - {task_id}"  # Mark as an actual task submission
                )
                
                # Show feedback
                if is_correct:
                    st.success(f"✅ 正确! {feedback_message}")
                    st.session_state.completed_tasks.add(task_id)
                    
                    # Check if all tasks are completed and update model selection availability
                    all_completed = all(st.session_state.task_completion[f"task{i}"]["correct"] for i in range(1, 6))
                    if all_completed:
                        st.session_state.can_select_model = True
                else:
                    # Get the task's concept hints
                    concept_hints = key_concepts[task_id]["concept_hints"]
                    
                    # Select 2 missing concepts to provide hints for (or fewer if less are missing)
                    num_hints = min(2, len(missing_concepts))
                    selected_missing = missing_concepts[:num_hints]
                    
                    # Create targeted feedback
                    hint_text = "请考虑包含以下关键要素: "
                    for concept in selected_missing:
                        hint_text += f"\n• {concept_hints[concept]}"
                    
                    st.error(f"❌ 还不完全正确。您的答案需要更多细节。 {hint_text}")
                
                # Stay on the RAG tab instead of switching back to About tab
                stay_on_rag_tab()
            
            # Handle skip button action
            if can_skip and skip_button:
                # Mark task as skipped
                st.session_state.skipped_tasks.add(task_id)
                
                # Log the skip action
                log_user_answer(
                    st.session_state.username, 
                    task_id, 
                    "SKIPPED", 
                    False,
                    model_used=st.session_state.task_models[task_id],
                    query=f"Task skipped - {task_id}"  # Mark as a skipped task
                )
                
                # Show confirmation
                st.warning(f"任务 {task_id[-1]} 已跳过。您可以继续其他任务。")
                
                # Check if all tasks are now completed or skipped
                if check_all_tasks_completed_or_skipped():
                    st.session_state.can_select_model = True
                    st.session_state.show_feedback_popup = True
                    st.success("所有任务已完成！您现在可以选择AI模型。")
                    # Stay on the RAG tab instead of switching back to About tab
                    stay_on_rag_tab()
                    st.rerun()  # Rerun to show the popup
            
            # Display task status summary
            st.divider()
            st.subheader("任务进度")
            
            for i in range(1, 6):
                task_key = f"task{i}"
                task_data = st.session_state.task_completion[task_key]
                
                if task_key in st.session_state.skipped_tasks:
                    status = "⏩ 已跳过"
                    color = "orange"
                elif not task_data["completed"]:
                    status = "⚪ 未尝试"
                    color = "gray"
                elif task_data["correct"]:
                    status = "✅ 已完成"
                    color = "green"
                else:
                    status = f"❌ 已尝试 ({task_data['attempts']})"
                    color = "red"
                
                st.markdown(f"**任务 {i}**: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

            # Check if all tasks are completed successfully or skipped
            if check_all_tasks_completed_or_skipped() and not st.session_state.can_select_model:
                st.session_state.can_select_model = True
                st.session_state.show_feedback_popup = True
                # Show message about model selection being available
                st.success("🎉 所有任务已完成！您现在可以选择AI模型。")

            # Show evaluation form if all tasks are completed successfully and evaluation not yet submitted
            if check_all_tasks_completed_or_skipped() and 'evaluation_submitted' not in st.session_state:
                st.session_state.show_evaluation_form = True

            # Display the evaluation form in a modal-like container
            if check_all_tasks_completed_or_skipped() and st.session_state.get('show_evaluation_form', False):
                st.markdown("### 🎉 恭喜完成所有任务！")
                
                with st.container():
                    st.markdown("""
                    ## 反馈表单
                    请告诉我们您对NASA经验教训系统的看法。
                    您的反馈将帮助我们改进系统。
                    """)
                    
                    with st.form("evaluation_form"):
                        # Part 1: System Usability Scale (SUS) Questions
                        st.header("系统易用性")
                        
                        # SUS Questions - Using 5-point Likert scale
                        sus_questions = [
                            "我认为我会经常使用这个系统。",
                            "我发现这个系统过于复杂。",
                            "我认为这个系统易于使用。",
                            "我认为需要技术人员的支持才能使用这个系统。",
                            "我发现这个系统的各项功能集成得很好。",
                            "我认为这个系统的不一致性太多。",
                            "我认为大多数人会很快学会使用这个系统。",
                            "我发现这个系统使用起来非常繁琐。",
                            "使用这个系统时我感到非常自信。",
                            "在开始使用这个系统之前，我需要学习很多东西。"
                        ]
                        
                        sus_responses = {}
                        for i, question in enumerate(sus_questions, 1):
                            sus_responses[f"sus_q{i}"] = st.radio(
                                question,
                                options=["非常不同意", "不同意", "中立", "同意", "非常同意"],
                                horizontal=True,
                                key=f"sus_q{i}"
                            )
                        
                        # Task difficulty questions
                        st.header("任务难度")
                        
                        # Create a more structured layout for task difficulty
                        task_difficulty = {}
                        for i in range(1, 6):
                            task_difficulty[f"task{i}"] = st.radio(
                                f"任务{i}有多难？",
                                options=["太简单", "简单", "刚好合适", "有挑战性", "太难"],
                                horizontal=True,
                                key=f"difficulty_task{i}"
                            )
                        
                        # AI Assistant Performance
                        st.header("AI助手性能")
                        
                        ai_helpfulness = st.radio(
                            "AI助手有多大帮助？",
                            options=["没有帮助", "稍微有帮助", "比较有帮助", "很有帮助", "非常有帮助"],
                            horizontal=True
                        )
                        
                        ai_relevance = st.radio(
                            "AI的回答与您的问题相关度如何？",
                            options=["不相关", "稍微相关", "比较相关", "很相关", "非常相关"],
                            horizontal=True
                        )
                        
                        # Research-Specific Questions
                        st.header("研究功能")
                        
                        retrieval_quality = st.radio(
                            "您如何评价找到的信息质量？",
                            options=["差", "一般", "好", "很好", "优秀"],
                            horizontal=True
                        )
                        
                        traditional_comparison = st.radio(
                            "与普通搜索方式相比，本系统:",
                            options=["差很多", "差一些", "差不多", "好一些", "好很多"],
                            horizontal=True
                        )
                        
                        # Open-ended feedback
                        st.header("其他反馈")
                        
                        improvement_suggestions = st.text_area(
                            "您认为如何改进本系统？",
                            height=100
                        )
                        
                        favorite_feature = st.text_area(
                            "您最喜欢的功能是什么？",
                            height=100
                        )
                        
                        # Submit button
                        submitted = st.form_submit_button("提交反馈")
                        
                        if submitted:
                            # Collect all evaluation data
                            evaluation_data = {
                                # System Usability Scale responses
                                "sus_responses": sus_responses,
                                
                                # Task difficulty
                                "task_difficulty": task_difficulty,
                                
                                # AI assistant evaluation
                                "ai_helpfulness": ai_helpfulness,
                                "ai_relevance": ai_relevance,
                                
                                # Research-specific evaluation
                                "retrieval_quality": retrieval_quality,
                                "traditional_comparison": traditional_comparison,
                                
                                # Open-ended feedback
                                "improvement_suggestions": improvement_suggestions,
                                "favorite_feature": favorite_feature
                            }
                            
                            # Log the evaluation
                            log_user_evaluation(st.session_state.username, evaluation_data)
                            
                            # Update session state
                            st.session_state.evaluation_submitted = True
                            st.session_state.show_evaluation_form = False
                            
                            # Show success message
                            st.success("感谢您的反馈！评估已成功提交。")
                            st.balloons()

    with tab1:
        st.header("NASA任务知识库")
        
        # 模型配置区域
        with st.expander("模型配置", expanded=not st.session_state.models_loaded):
            # 1. 选择模型提供商
            provider = st.selectbox(
                "选择模型提供商",
                ["OpenAI", "Anthropic", "本地模型"],
                index=["OpenAI", "Anthropic", "本地模型"].index(st.session_state.selected_provider),
                key="provider_select"
            )
            st.session_state.selected_provider = provider
            
            # 2. 根据提供商显示配置选项（按顺序排列）
            if provider == "OpenAI":
                api_key = st.text_input(
                    "API Key",
                    value=st.session_state.provider_config["OpenAI"].get("api_key", ""),
                    type="password",
                    placeholder="sk-..."
                )
                base_url = st.text_input(
                    "Base URL",
                    value=st.session_state.provider_config["OpenAI"].get("base_url", "https://api.openai.com/v1")
                )
                st.session_state.provider_config["OpenAI"]["api_key"] = api_key
                st.session_state.provider_config["OpenAI"]["base_url"] = base_url
            
            elif provider == "Anthropic":
                api_key = st.text_input(
                    "API Key",
                    value=st.session_state.provider_config["Anthropic"].get("api_key", ""),
                    type="password",
                    placeholder="sk-ant-..."
                )
                base_url = st.text_input(
                    "Base URL",
                    value=st.session_state.provider_config["Anthropic"].get("base_url", "https://api.anthropic.com")
                )
                st.session_state.provider_config["Anthropic"]["api_key"] = api_key
                st.session_state.provider_config["Anthropic"]["base_url"] = base_url
            
            elif provider == "本地模型":
                base_url = st.text_input(
                    "服务地址",
                    value=st.session_state.provider_config["本地模型"].get("base_url", "http://localhost:8000/v1")
                )
                api_key = st.text_input(
                    "API Key（可选）",
                    value=st.session_state.provider_config["本地模型"].get("api_key", "not-needed"),
                    type="password",
                    placeholder="留空则不需要"
                )
                st.session_state.provider_config["本地模型"]["base_url"] = base_url
                st.session_state.provider_config["本地模型"]["api_key"] = api_key
            
            # 3. 保存并加载按钮
            if st.button("保存并加载模型", type="primary", use_container_width=True):
                config = st.session_state.provider_config[provider]
                models, error = load_models_from_provider(provider, config)
                st.session_state.loaded_models = models
                st.session_state.model_load_error = error
                st.session_state.models_loaded = len(models) > 0
                
                if models:
                    st.session_state.selected_model_name = models[0]  # 默认选中第一个
                    st.session_state.use_custom_model = False
                    st.success(f"成功加载 {len(models)} 个模型")
                else:
                    st.session_state.use_custom_model = True
                    if error:
                        st.warning(f"加载失败: {error}，已切换到自定义模式")
                    else:
                        st.warning("未找到可用模型，已切换到自定义模式")
                
                # 保存配置到本地文件
                save_config_to_file()
                st.rerun()
        
        # 4. 模型选择区域
        st.subheader("选择模型")
        
        if st.session_state.models_loaded and st.session_state.loaded_models:
            # 显示模型选择下拉框
            model_options = st.session_state.loaded_models + ["自定义模型"]
            
            # 确定默认选中的索引
            if st.session_state.use_custom_model:
                default_index = len(model_options) - 1  # 选中"自定义模型"
            else:
                try:
                    default_index = model_options.index(st.session_state.selected_model_name)
                except ValueError:
                    default_index = 0
            
            selected = st.selectbox(
                "选择模型",
                model_options,
                index=default_index,
                label_visibility="collapsed",
                key="model_selector"
            )
            
            if selected == "自定义模型":
                st.session_state.use_custom_model = True
                custom_name = st.text_input(
                    "输入模型名称",
                    value=st.session_state.custom_model_name,
                    placeholder="例如: gpt-4, claude-3-opus, llama-3-70b",
                    key="custom_model_input"
                )
                st.session_state.custom_model_name = custom_name
                current_model = custom_name
            else:
                st.session_state.use_custom_model = False
                st.session_state.selected_model_name = selected
                current_model = selected
            
            # 保存配置
            save_config_to_file()
            
            # 显示当前选中的模型
            st.info(f"当前使用模型: **{current_model}**")
        
        else:
            # 未加载模型时显示自定义输入
            st.session_state.use_custom_model = True
            custom_name = st.text_input(
                "输入模型名称",
                value=st.session_state.custom_model_name,
                placeholder="例如: gpt-4, claude-3-opus, llama-3-70b",
                key="custom_model_input_fallback"
            )
            st.session_state.custom_model_name = custom_name
            
            # 保存配置
            save_config_to_file()
            
            if custom_name:
                st.info(f"当前使用模型: **{custom_name}**")
            else:
                st.warning("请配置模型提供商或输入自定义模型名称")
        
        # Chat interface for RAG
        if 'rag_messages' not in st.session_state:
            st.session_state.rag_messages = [
                {"role": "assistant", "content": "我可以回答关于NASA任务和文档的问题。请问您想了解什么？"}
            ]
        
        # Display chat messages
        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # User input
        if rag_query := st.chat_input("输入关于NASA任务的问题...", key="rag_input"):
            # Add user message to chat history
            st.session_state.rag_messages.append({"role": "user", "content": rag_query})
            
            # Display user message
            with st.chat_message("user"):
                st.write(rag_query)
            
            # Use the current task ID from session state instead of trying to determine it here
            current_task_id = st.session_state.current_task_id
            
            # 获取当前使用的模型名称
            display_model = get_current_model_name()
            
            # Get response from RAG model
            with st.spinner(f"正在使用 {display_model} 思考中..."):
                response = process_rag_query(rag_query, current_task_id)
            
            # Add assistant response to chat history
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)

# Display feedback form as a popup when all tasks are completed/skipped
if st.session_state.show_feedback_popup and 'evaluation_submitted' not in st.session_state:
    # Create a custom dialog-like interface instead of using st.dialog()
    feedback_container = st.container()
    
    with feedback_container:
        # Add a colored background container to make it stand out
        with st.container(border=True):
            st.markdown("## 反馈表单 - 请在继续前完成")
            st.markdown("### 感谢完成所有任务！")
            
            with st.form("popup_evaluation_form"):
                # Part 1: System Usability Scale (SUS) Questions
                st.header("系统易用性")
                
                # SUS Questions - Using 5-point Likert scale
                sus_questions = [
                    "我认为我会经常使用这个系统。",
                    "我发现这个系统过于复杂。",
                    "我认为这个系统易于使用。",
                    "我认为需要技术人员的支持才能使用这个系统。",
                    "我发现这个系统的各项功能集成得很好。",
                    "我认为这个系统的不一致性太多。",
                    "我认为大多数人会很快学会使用这个系统。",
                    "我发现这个系统使用起来非常繁琐。",
                    "使用这个系统时我感到非常自信。",
                    "在开始使用这个系统之前，我需要学习很多东西。"
                ]
                
                sus_responses = {}
                for i, question in enumerate(sus_questions, 1):
                    sus_responses[f"sus_q{i}"] = st.radio(
                        question,
                        options=["非常不同意", "不同意", "中立", "同意", "非常同意"],
                        horizontal=True,
                        key=f"popup_sus_q{i}"
                    )
                
                # Task difficulty questions
                st.header("任务难度")
                
                # Create a more structured layout for task difficulty
                task_difficulty = {}
                for i in range(1, 6):
                    task_key = f"task{i}"
                    if task_key in st.session_state.skipped_tasks:
                        # If task was skipped, mark as "Too Difficult" by default, but allow changing
                        difficulty_options = ["太简单", "简单", "刚好合适", "有挑战性", "太难"]
                        default_idx = 4  # "Too Difficult"
                        task_difficulty[task_key] = st.radio(
                            f"任务{i}有多难？（已跳过）",
                            options=difficulty_options,
                            index=default_idx,
                            horizontal=True,
                            key=f"popup_difficulty_task{i}"
                        )
                    else:
                        task_difficulty[task_key] = st.radio(
                            f"任务{i}有多难？",
                            options=["太简单", "简单", "刚好合适", "有挑战性", "太难"],
                            horizontal=True,
                            key=f"popup_difficulty_task{i}"
                        )
                
                # AI Assistant Performance
                st.header("AI助手性能")
                
                ai_helpfulness = st.radio(
                    "AI助手有多大帮助？",
                    options=["没有帮助", "稍微有帮助", "比较有帮助", "很有帮助", "非常有帮助"],
                    horizontal=True,
                    key="popup_ai_helpfulness"
                )
                
                ai_relevance = st.radio(
                    "AI的回答与您的问题相关度如何？",
                    options=["不相关", "稍微相关", "比较相关", "很相关", "非常相关"],
                    horizontal=True,
                    key="popup_ai_relevance"
                )
                
                # Research-Specific Questions
                st.header("研究功能")
                
                retrieval_quality = st.radio(
                    "您如何评价找到的信息质量？",
                    options=["差", "一般", "好", "很好", "优秀"],
                    horizontal=True,
                    key="popup_retrieval_quality"
                )
                
                traditional_comparison = st.radio(
                    "与普通搜索方式相比，本系统:",
                    options=["差很多", "差一些", "差不多", "好一些", "好很多"],
                    horizontal=True,
                    key="popup_traditional_comparison"
                )
                
                # Open-ended feedback
                st.header("其他反馈")
                
                improvement_suggestions = st.text_area(
                    "您认为如何改进本系统？",
                    height=100,
                    key="popup_improvement_suggestions"
                )
                
                favorite_feature = st.text_area(
                    "您最喜欢的功能是什么？",
                    height=100,
                    key="popup_favorite_feature"
                )
                
                # Add skipped tasks information
                skipped_tasks_list = list(st.session_state.skipped_tasks)
                
                # Submit button
                submitted = st.form_submit_button("提交反馈")
                
                if submitted:
                    # Collect all evaluation data
                    evaluation_data = {
                        # System Usability Scale responses
                        "sus_responses": sus_responses,
                        
                        # Task difficulty
                        "task_difficulty": task_difficulty,
                        
                        # AI assistant evaluation
                        "ai_helpfulness": ai_helpfulness,
                        "ai_relevance": ai_relevance,
                        
                        # Research-specific evaluation
                        "retrieval_quality": retrieval_quality,
                        "traditional_comparison": traditional_comparison,
                        
                        # Open-ended feedback
                        "improvement_suggestions": improvement_suggestions,
                        "favorite_feature": favorite_feature,
                        
                        # Add information about skipped tasks
                        "skipped_tasks": skipped_tasks_list
                    }
                    
                    # Log the evaluation
                    log_user_evaluation(st.session_state.username, evaluation_data)
                    
                    # Update session state
                    st.session_state.evaluation_submitted = True
                    st.session_state.show_feedback_popup = False
                    
                    # Show success message
                    st.success("感谢您的反馈！评估已成功提交。")
                    st.balloons()
                    
                    # Rerun to close the feedback form
                    st.rerun()