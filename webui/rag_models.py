import os
import time
import logging
from datetime import datetime
import json
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional
import requests
import openai
import threading
from supabase import create_client, Client

# Add Together AI integration
try:
    from together import Together
    TOGETHER_AVAILABLE = True
    print("TogetherAI library successfully imported!")
except ImportError as e:
    TOGETHER_AVAILABLE = False
    print(f"TogetherAI library not available: {e}. Will fall back to OpenAI.")

# Load environment variables from .env file if it exists
load_dotenv()

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
    print("Supabase connection successful in rag_models.py!")
except Exception as e:
    st.session_state.supabase_connected = False
    print(f"Failed to connect to Supabase in rag_models.py: {e}")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"rag_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom logger handler to also log to Supabase
class SupabaseHandler(logging.Handler): 
    def emit(self, record):
        if not hasattr(st.session_state, 'supabase_connected') or not st.session_state.supabase_connected:
            return
        
        try:
            # Check if table exists
            try:
                supabase.table("system_logs").select("*").limit(1).execute()
                table_exists = True
            except Exception:
                # Table doesn't exist - silently fail
                print("system_logs table doesn't exist in Supabase. Skipping remote logging.")
                return
            
            if table_exists:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "message": self.format(record),
                    "logger": record.name,
                    "pathname": record.pathname,
                    "lineno": record.lineno
                }
                
                # Insert into Supabase
                supabase.table("system_logs").insert(log_entry).execute()
        except Exception as e:
            # Don't use logger here to avoid infinite recursion
            print(f"Failed to log to Supabase: {e}")

# Add Supabase handler if connected
if supabase and st.session_state.get('supabase_connected', False):
    supabase_handler = SupabaseHandler()
    supabase_handler.setLevel(logging.INFO)
    logger.addHandler(supabase_handler)

# Try to get API keys from environment variables first
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')  # Add Together API key

# If not found in environment, try to get from Streamlit secrets
try:
    if not openai_api_key and 'OPENAI_API_KEY' in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']
    if not langchain_api_key and 'LANGCHAIN_API_KEY' in st.secrets:
        langchain_api_key = st.secrets['LANGCHAIN_API_KEY']
    if not together_api_key and 'TOGETHER_API_KEY' in st.secrets:
        together_api_key = st.secrets['TOGETHER_API_KEY']
except Exception as e:
    logger.warning(f"Could not access Streamlit secrets: {e}")

# Set the API keys if found
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
if together_api_key:
    os.environ['TOGETHER_API_KEY'] = together_api_key

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize cache
set_llm_cache(InMemoryCache())

# Define the NASA documents prompt template with source citation
nasa_prompt = ChatPromptTemplate.from_template("""
You are an expert in analyzing NASA documents and mission data. Based on the following context, please provide concise answers derived from the context.

Context: {context}

Question: {input}

Format:
1. Brief Answer
2. Key Points
3. Relevant Mission Details (if applicable)

DO NOT include a Sources section in your response. The system will add this automatically based on the actual documents used.
""")

# Create a custom langchain LLM class for TogetherAI
# Going for unstructured approach of having several classes in a single file, not the best practice, but had a lot of issues/bugs when running, so having everything in one file made it easier to track issues.
# Move the classes into own files later on.
class TogetherLLM(LLM):
    """Custom LLM class for TogetherAI that inherits from LangChain's LLM base class."""
    
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    temperature: float = 0.1
    max_tokens: int = 1024
    together_client: Any = None
    
    def __init__(self, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.1, max_tokens=1024):
        """Initialize TogetherLLM."""
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.together_client = Together()
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together_ai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Process the prompt using TogetherAI."""
        try:
            response = self.together_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error using TogetherAI: {e}")
            return f"Error: Could not generate response with TogetherAI ({str(e)})"

class RAGModel:
    def __init__(self, data_dir=None, index_dir=None):
        # Get the absolute path of the current file
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent  # webui的上级目录即项目根目录
        
        # Simple_RAG的数据和索引路径
        self.data_dir = Path(data_dir) if data_dir else project_root / "data" / "nasa"
        self.index_dir = Path(index_dir) if index_dir else project_root / "vector_index"
        
        self.chunks_dirs = [self.data_dir]
        self.lessons_learned_path = None

        self.db = None
        self.retriever = None
        self.llm = None
        self.together_llm = None
        self.model_loading = False
        self.model_loaded = False
        self.together_available = TOGETHER_AVAILABLE and together_api_key is not None
        
        # Initialize embeddings model - 使用与索引构建相同的模型
        self.embed = HuggingFaceEmbeddings(
            model_name="microsoft/harrier-oss-v1-270m",
            model_kwargs={'device': 'cpu'}
        )
        
        # Start model loading in background
        self._start_background_model_loading()
        
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"索引目录: {self.index_dir}")
        
        # Load or create vector store
        self._load_or_create_vector_store()
    
    def _start_background_model_loading(self):
        """Start loading the Mistral model in a background thread"""
        def load_model():
            logger.info("Starting background model loading...")
            self.model_loading = True
            try:
                # MODIFIED: Try TogetherAI first if available
                if self.together_available:
                    logger.info("Trying TogetherAI first...")
                    try:
                        self.together_llm = TogetherLLM()
                        # Test with a simple call
                        test_response = self.together_llm.invoke("Hello")
                        if test_response and not test_response.startswith("Error:"):
                            self.model_loaded = True
                            logger.info("Using TogetherAI Llama-3.3-70B model for responses")
                            return  # Successfully loaded TogetherAI, no need to try Ollama
                        else:
                            logger.warning(f"TogetherAI test call failed with response: {test_response}")
                            raise Exception("TogetherAI test failed")
                    except Exception as e:
                        logger.warning(f"TogetherAI not available: {str(e)}")
                        self.together_llm = None  # Reset to None in case it was partially initialized
                else:
                    logger.warning("TogetherAI not available: library not imported or API key missing")
                
                # WE ARE NOT LONGER USING THE wizardlm2 model, but for it is kept as an possibility when testing locally in the future.
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        self.llm = OllamaLLM(
                            model="wizardlm2",
                            temperature=0.1,
                            num_ctx=512,
                            request_timeout=60.0,
                            num_predict=256,
                            num_thread=4,
                            stop=["4. Sources"]
                        )
                        # Make a dummy call to ensure model is loaded
                        self.llm.invoke("Hello")
                        self.model_loaded = True
                        logger.info("Using Ollama model (wizardlm2) for responses as fallback")
                    else:
                        logger.warning(f"Ollama API returned status code: {response.status_code}")
                        raise Exception("Ollama not available")
                except Exception as ollama_error:
                    logger.warning(f"Ollama not available: {str(ollama_error)}")
                    raise Exception(f"Neither TogetherAI nor Ollama are available. TogetherAI error: {str(e) if 'e' in locals() else 'Not attempted'}, Ollama error: {str(ollama_error)}")
            except Exception as e:
                logger.error(f"Error loading local models: {str(e)}")
                logger.warning("Falling back to OpenAI")
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=1024
                )
                self.model_loaded = True
                logger.info("Using OpenAI model (gpt-4o-mini) for responses")
            finally:
                self.model_loading = False
        
        # Start the loading thread
        thread = threading.Thread(target=load_model)
        thread.daemon = True  # Thread will be killed when main program exits
        thread.start()
    
    def _load_chunks(self):
        """Load document chunks from JSON files and CSV files"""
        logger.info(f"Loading document chunks from multiple sources...")
        chunks = []
        
        # Process JSON files from all directories
        for chunks_dir in self.chunks_dirs:
            logger.info(f"Processing directory: {chunks_dir}")
            # Get all JSON files in the chunks directory
            json_files = list(chunks_dir.glob("**/*.json*"))
            
            if not json_files:
                logger.warning(f"No JSON files found in {chunks_dir}")
                # List directory contents to help debug
                try:
                    logger.info(f"Directory contents: {list(chunks_dir.iterdir())}")
                except Exception as e:
                    logger.error(f"Could not list directory contents: {e}")
                continue
                
            logger.info(f"Found {len(json_files)} JSON files in {chunks_dir}")
            
            # Process each JSON file
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        # Handle both single objects and arrays
                        if isinstance(data, list):
                            items = data
                        else:
                            items = [data]
                        
                        for item in items:
                            if 'content' in item and 'title' in item:
                                # Extract metadata properly
                                metadata_dict = {
                                    'title': item['title'],
                                    'chunk_id': item.get('chunk_id', ''),
                                    'page_number': item.get('page_number', ''),
                                    'section_level': item.get('section_level', ''),
                                    'file_name': item.get('metadata', {}).get('file_name', ''),
                                    'source_type': 'pdf'
                                }
                                
                                # Add download_url directly to the metadata
                                if 'metadata' in item and 'download_url' in item['metadata']:
                                    metadata_dict['download_url'] = item['metadata']['download_url']
                                
                                # Create document with content and metadata
                                doc = Document(
                                    page_content=item['content'],
                                    metadata=metadata_dict
                                )
                                chunks.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        
        # Process NASA Lessons Learned CSV file
        if self.lessons_learned_path.exists():
            logger.info(f"Processing NASA Lessons Learned from {self.lessons_learned_path}")
            try:
                import pandas as pd
                df = pd.read_csv(self.lessons_learned_path)
                
                # Process each row in the CSV
                for _, row in df.iterrows():
                    # Skip header row if it got included
                    if row.get('url') == 'url':
                        continue
                        
                    # Combine relevant fields into content
                    content_parts = []
                    
                    if not pd.isna(row.get('subject')):
                        content_parts.append(f"Subject: {row['subject']}")
                    
                    if not pd.isna(row.get('abstract')) and row['abstract'] != 'None':
                        content_parts.append(f"Abstract: {row['abstract']}")
                    
                    if not pd.isna(row.get('driving_event')) and row['driving_event'] != 'None':
                        content_parts.append(f"Driving Event: {row['driving_event']}")
                    
                    if not pd.isna(row.get('lessons_learned')) and row['lessons_learned'] != 'None':
                        content_parts.append(f"Lessons Learned: {row['lessons_learned']}")
                    
                    if not pd.isna(row.get('recommendations')) and row['recommendations'] != 'None':
                        content_parts.append(f"Recommendations: {row['recommendations']}")

                    if not pd.isna(row.get('evidence')) and row['evidence'] != 'None':
                        content_parts.append(f"Evidence: {row['evidence']}")

                    if not pd.isna(row.get('program_relation')) and row['program_relation'] != 'None':
                        content_parts.append(f"Program Relation: {row['program_relation']}")

                    if not pd.isna(row.get('program_phase')) and row['program_phase'] != 'None':
                        content_parts.append(f"Program Phase: {row['program_phase']}")

                    if not pd.isna(row.get('mission_directorate')) and row['mission_directorate'] != 'None':
                        content_parts.append(f"Mission Directorate: {row['mission_directorate']}")

                    if not pd.isna(row.get('topics')) and row['topics'] != 'None':
                        content_parts.append(f"Topics: {row['topics']}")
                    
                    content = "\n\n".join(content_parts)
                    
                    # Create metadata
                    metadata_dict = {
                        'title': row.get('subject', 'NASA Lesson Learned'),
                        'url': row.get('url', ''),
                        'source_type': 'lessons_learned',
                        'mission_directorate': row.get('mission_directorate', ''),
                        'program_phase': row.get('program_phase', ''),
                        'topics': row.get('topics', '')
                    }
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata=metadata_dict
                    )
                    chunks.append(doc)
                    
            except Exception as e:
                logger.error(f"Error processing NASA Lessons Learned CSV: {e}")
        
        logger.info(f"Loaded {len(chunks)} total document chunks")
        return chunks
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one"""
        # Simple_RAG使用 index.faiss 和 index.pkl
        index_faiss = self.index_dir / "index.faiss"
        index_pkl = self.index_dir / "index.pkl"
        
        if index_faiss.exists() and index_pkl.exists():
            logger.info("加载已有向量索引...")
            try:
                self.db = FAISS.load_local(
                    str(self.index_dir),
                    self.embed,
                    allow_dangerous_deserialization=True
                )
                logger.info("向量索引加载成功")
            except Exception as e:
                logger.error(f"加载向量索引失败: {e}")
                logger.info("尝试创建新索引...")
                self._create_vector_store()
        else:
            logger.info("未找到索引文件，尝试创建新索引...")
            self._create_vector_store()
        
        # Create retriever
        if self.db:
            self.retriever = self.db.as_retriever(
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.7
                }
            )
    def _create_vector_store(self):
        """Create a new vector store from document chunks"""
        chunks = self._load_chunks()
        
        if not chunks:
            logger.warning("No chunks to create vector store")
            return
        
        # Create vector store with batching
        logger.info("Creating vector store...")
        BATCH_SIZE = 32
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(total_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(chunks))
            batch = chunks[start_idx:end_idx]
            
            batch_start_time = time.time()
            
            if i == 0:
                self.db = FAISS.from_documents(batch, self.embed)
                print("9")
            else:
                self.db.add_documents(batch)
                print("10")
            batch_duration = time.time() - batch_start_time
            docs_per_second = len(batch) / batch_duration
            logger.info(f"Batch {i+1}/{total_batches} completed in {batch_duration:.2f} seconds ({docs_per_second:.2f} docs/second)")
        
        # Save the vector store
        if self.db:
            self.db.save_local(str(self.index_dir / "nasa_docs_index"))
            logger.info("Vector store created and saved successfully")
    
    def query(self, question, provider="OpenAI", model_name="gpt-4o-mini", config=None):
        """Query the RAG model with a question"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return "Retriever not initialized. Please try again later."
        
        if config is None:
            config = {}
        
        logger.info(f"Processing question with {provider}/{model_name}: {question}")
        logger.info(f"Config: api_key={'***' if config.get('api_key') else 'empty'}, base_url={config.get('base_url', 'default')}")
        query_start_time = time.time()
        
        try:
            # Retrieve documents
            self.retriever.search_kwargs["k"] = 3
            
            # Get documents
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Print document metadata for debugging
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Document {i+1} metadata: {doc.metadata}")
            
            # Initialize the appropriate LLM based on provider
            llm = None
            
            if provider == "OpenAI":
                from langchain_openai import ChatOpenAI
                api_key = config.get("api_key", "") or os.getenv('OPENAI_API_KEY', '')
                base_url = config.get("base_url", "https://api.openai.com/v1")
                logger.info(f"OpenAI config: base_url={base_url}, api_key={'***' if api_key else 'empty'}")
                if not api_key:
                    return "请先设置OpenAI API Key"
                try:
                    llm = ChatOpenAI(
                        model=model_name,
                        openai_api_key=api_key,
                        openai_api_base=base_url,
                        temperature=0.1,
                        max_tokens=1024
                    )
                    # 测试LLM是否可用
                    test_result = llm.invoke("Hello, respond with OK")
                    logger.info(f"LLM测试成功: {test_result.content[:50]}")
                except Exception as llm_error:
                    logger.error(f"创建或测试LLM失败: {llm_error}")
                    return f"LLM初始化失败: {str(llm_error)}"
                logger.info(f"Using OpenAI model: {model_name}")
            
            elif provider == "Anthropic":
                try:
                    from langchain_anthropic import ChatAnthropic
                    api_key = config.get("api_key", "") or os.getenv('ANTHROPIC_API_KEY', '')
                    base_url = config.get("base_url", "https://api.anthropic.com")
                    if not api_key:
                        return "请先设置Anthropic API Key"
                    llm = ChatAnthropic(
                        model=model_name,
                        anthropic_api_key=api_key,
                        anthropic_api_url=base_url,
                        temperature=0.1,
                        max_tokens=1024
                    )
                    logger.info(f"Using Anthropic model: {model_name}")
                except ImportError:
                    return "请安装 langchain-anthropic: pip install langchain-anthropic"
                except Exception as e:
                    logger.error(f"Anthropic LLM初始化失败: {e}")
                    return f"Anthropic LLM初始化失败: {str(e)}"
            
            elif provider == "本地模型":
                # 尝试使用TogetherAI（如果模型名包含llama）
                if "llama" in model_name.lower() and self.together_available:
                    if not self.together_llm:
                        try:
                            self.together_llm = TogetherLLM(model_name=model_name)
                            test_response = self.together_llm.invoke("Hello")
                            if test_response.startswith("Error:"):
                                self.together_llm = None
                        except:
                            self.together_llm = None
                    
                    if self.together_llm:
                        llm = self.together_llm
                        logger.info(f"Using TogetherAI model: {model_name}")
                
                # 如果没有TogetherAI，尝试使用本地vLLM/OpenAI兼容API
                if llm is None:
                    from langchain_openai import ChatOpenAI
                    base_url = config.get("base_url", "http://localhost:8000/v1")
                    api_key = config.get("api_key", "not-needed")
                    try:
                        llm = ChatOpenAI(
                            model=model_name,
                            openai_api_key=api_key,
                            openai_api_base=base_url,
                            temperature=0.1,
                            max_tokens=1024
                        )
                        # 测试LLM
                        test_result = llm.invoke("Hello")
                        logger.info(f"本地模型测试成功")
                    except Exception as e:
                        logger.error(f"本地模型初始化失败: {e}")
                        return f"本地模型连接失败: {str(e)}"
                    logger.info(f"Using local model: {model_name} at {base_url}")
            
            if llm is None:
                return "无法初始化LLM，请检查配置。"
            
            # Use NASA prompt
            combine_docs_chain = create_stuff_documents_chain(llm, nasa_prompt)
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                self.retriever,
                combine_docs_chain
            )
            
            # Execute the query
            try:
                result = retrieval_chain.invoke(
                    {'input': question},
                    config={"timeout": 60}
                )
                answer = result["answer"]
            except Exception as e:
                logger.error(f"Query timeout or error: {type(e).__name__}: {e}")
                return f"查询超时或出错: {str(e)}"
            
            # Add source information
            sources = []
            for doc in retrieved_docs:
                source_type = doc.metadata.get("source_type", "unknown")
                
                if source_type == "pdf":
                    file_name = doc.metadata.get("file_name", "Unknown")
                    download_url = doc.metadata.get("download_url", "")
                    if download_url:
                        sources.append(f"- {file_name}: {download_url}")
                    else:
                        sources.append(f"- {file_name}")
                
                elif source_type == "lessons_learned":
                    url = doc.metadata.get("url", "")
                    title = doc.metadata.get("title", "NASA Lesson Learned")
                    sources.append(f"- {title}: {url}")
            
            if sources:
                answer += "\n\n**来源:**\n" + "\n".join(sources)
            
            query_duration = time.time() - query_start_time
            logger.info(f"Query completed in {query_duration:.2f} seconds")
            
            return answer
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing query: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"处理查询时出错 ({type(e).__name__}): {str(e)}"

    def get_document_by_id(self, doc_id):
        logger.info(f"Looking for document with ID: {doc_id}")
        
        # First check if it's a lessons learned URL
        if doc_id.startswith("https://"):
            if self.lessons_learned_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(self.lessons_learned_path)
                    lesson = df[df['url'] == doc_id].iloc[0]
                    
                    # Combine relevant fields into content
                    content_parts = []
                    if not pd.isna(lesson.get('subject')):
                        content_parts.append(f"Subject: {lesson['subject']}")
                    if not pd.isna(lesson.get('abstract')) and lesson['abstract'] != 'None':
                        content_parts.append(f"Abstract: {lesson['abstract']}")
                    if not pd.isna(lesson.get('driving_event')) and lesson['driving_event'] != 'None':
                        content_parts.append(f"Driving Event: {lesson['driving_event']}")
                    if not pd.isna(lesson.get('lessons_learned')) and lesson['lessons_learned'] != 'None':
                        content_parts.append(f"Lessons Learned: {lesson['lessons_learned']}")
                    if not pd.isna(lesson.get('recommendations')) and lesson['recommendations'] != 'None':
                        content_parts.append(f"Recommendations: {lesson['recommendations']}")
                    if not pd.isna(lesson.get('evidence')) and lesson['evidence'] != 'None':
                        content_parts.append(f"Evidence: {lesson['evidence']}")
                    if not pd.isna(lesson.get('program_relation')) and lesson['program_relation'] != 'None':
                        content_parts.append(f"Program Relation: {lesson['program_relation']}")
                    if not pd.isna(lesson.get('program_phase')) and lesson['program_phase'] != 'None':
                        content_parts.append(f"Program Phase: {lesson['program_phase']}")
                    if not pd.isna(lesson.get('mission_directorate')) and lesson['mission_directorate'] != 'None':
                        content_parts.append(f"Mission Directorate: {lesson['mission_directorate']}")
                    if not pd.isna(lesson.get('topics')) and lesson['topics'] != 'None':
                        content_parts.append(f"Topics: {lesson['topics']}")
                    
                    content = "\n\n".join(content_parts)
                    
                    # Create metadata
                    metadata_dict = {
                        'title': lesson.get('subject', 'NASA Lesson Learned'),
                        'url': doc_id,
                        'source_type': 'lessons_learned',
                        'mission_directorate': lesson.get('mission_directorate', ''),
                        'program_phase': lesson.get('program_phase', ''),
                        'topics': lesson.get('topics', '')
                    }
                    
                    return Document(
                        page_content=content,
                        metadata=metadata_dict
                    )
                except Exception as e:
                    logger.error(f"Error retrieving lesson learned document: {e}")
                    return None
        
        # If not a URL, treat as a PDF chunk ID
        for chunks_dir in self.chunks_dirs:
            try:
                # The chunk ID format is typically "filename_pagenumber"
                # Extract the filename part
                file_prefix = doc_id.split('_')[0] if '_' in doc_id else doc_id
                
                # Search through all JSON files in the directory
                for json_file in chunks_dir.glob("**/*.json*"):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        # Handle both single objects and arrays
                        items = data if isinstance(data, list) else [data]
                        
                        for item in items:
                            if item.get('chunk_id') == doc_id:
                                metadata_dict = {
                                    'title': item.get('title', ''),
                                    'chunk_id': doc_id,
                                    'page_number': item.get('page_number', ''),
                                    'section_level': item.get('section_level', ''),
                                    'file_name': item.get('metadata', {}).get('file_name', ''),
                                    'source_type': 'pdf'
                                }
                                
                                # Add download_url if available
                                if 'metadata' in item and 'download_url' in item['metadata']:
                                    metadata_dict['download_url'] = item['metadata']['download_url']
                                
                                return Document(
                                    page_content=item.get('content', ''),
                                    metadata=metadata_dict
                                )
            except Exception as e:
                logger.error(f"Error searching in {chunks_dir}: {e}")
                continue
        
        logger.warning(f"Document with ID {doc_id} not found")
        return None

# Singleton instance
_rag_model = None

def get_rag_model():
    """Get or create the RAG model singleton"""
    global _rag_model
    if _rag_model is None:
        _rag_model = RAGModel()
    return _rag_model

if __name__ == "__main__":
    # Test the RAG model
    rag = get_rag_model()
    question = "What can we learn from the Gateway mission?"
    
    print("\nTesting with OpenAI:")
    answer_openai = rag.query(question, "openai")
    print(f"\nAnswer (OpenAI): {answer_openai}")
    
    print("\nTesting with Llama:")
    answer_llama = rag.query(question, "llama")
    print(f"\nAnswer (Llama): {answer_llama}") 