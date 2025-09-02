"""
Document Intelligence RAG Pipeline with Docling Processing

This Kubeflow pipeline implements an advanced Document Intelligence RAG system that:
1. Processes complex academic documents using Docling's intelligent document processing
2. Extracts and preserves tables, formulas, figures, and document structure
3. Creates enhanced RAG system with semantic search capabilities
4. Tests document intelligence queries on complex academic content
"""

import kfp
from typing import NamedTuple, Optional, List, Dict, Any
from kfp import dsl, components
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Artifact
)

# =============================================================================
# COMPONENT 1: DOCUMENT INTELLIGENCE SETUP
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=[
        'llama_stack_client', 
        'requests'
    ]
)
def docling_setup_component(
    embedding_model: str,
    embedding_dimension: int,
    chunk_size_tokens: int,
    vector_provider: str,
    docling_service: str,
    processing_timeout: int,
    llama_stack_url: str,
    model_id: str,
    temperature: float,
    max_tokens: int
) -> NamedTuple("SetupOutput", [("setup_config", Dict[str, Any])]):
    """
    Initialize the Document Intelligence RAG system with LlamaStack client and model configuration.
    
    Args:
        embedding_model: Sentence transformer model for text embeddings
        embedding_dimension: Vector dimensions (must match the embedding model)
        chunk_size_tokens: Optimal chunk size for academic content processing
        vector_provider: Vector database backend provider (e.g., "milvus")
        docling_service: URL of the Docling document processing service
        processing_timeout: Timeout in seconds for complex document processing
        llama_stack_url: URL of the LlamaStack service
        model_id: Model identifier for text generation
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens for model responses
    
    Returns:
        NamedTuple containing setup configuration for downstream components
        NamedTuple usage: https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/parameters/#multiple-output-parameters
    """
    import uuid
    from collections import namedtuple
    
    print("Initializing Document Intelligence RAG System")
    print("=" * 60)
    
    # LlamaStack configuration
    base_url = llama_stack_url
    
    # Model configuration
    model_config = {
        "model_id": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    # Configure sampling strategy for consistent, factual document analysis
    if model_config["temperature"] > 0.0:
        sampling_strategy = {
            "type": "top_p", 
            "temperature": model_config["temperature"], 
            "top_p": 0.95
        }
    else:
        sampling_strategy = {"type": "greedy"}    # Deterministic for factual analysis
    
    # Package parameters for LlamaStack inference API
    sampling_params = {
        "strategy": sampling_strategy,
        "max_tokens": model_config["max_tokens"],
    }
    
    # Document intelligence configuration
    document_intelligence_config = {
        "embedding_model": embedding_model,       # Sentence transformer for embeddings
        "embedding_dimension": embedding_dimension, # Vector dimensions (must match model)
        "chunk_size_tokens": chunk_size_tokens,   # Optimal chunk size for academic content
        "vector_provider": vector_provider,       # Use Milvus as vector store backend
        "docling_service": docling_service,       # Docling document processing service URL
        "processing_timeout": processing_timeout  # Timeout for complex documents
    }
    
    # Combine all configuration
    setup_config = {
        "base_url": base_url,
        "model_config": model_config,
        "sampling_params": sampling_params,
        "document_intelligence": document_intelligence_config,
        "vector_db_id": f"docling_vector_db_{uuid.uuid4()}"  # Unique identifier
    }
    
    print(f"Document Intelligence Setup Complete:")
    print(f"  • LlamaStack URL: {base_url}")
    print(f"  • Model: {model_config['model_id']}")
    print(f"  • Strategy: {sampling_strategy['type']}")
    print(f"  • Max Tokens: {model_config['max_tokens']}")
    print(f"  • Embedding Model: {document_intelligence_config['embedding_model']}")
    print(f"  • Vector Database ID: {setup_config['vector_db_id']}")
    print(f"  • Docling Service: {document_intelligence_config['docling_service']}")
    print("Ready for intelligent document processing!")
    
    # Return configuration for downstream components
    SetupOutput = namedtuple("SetupOutput", ["setup_config"])
    return SetupOutput(setup_config=setup_config)

# =============================================================================
# COMPONENT 2: DOCLING DOCUMENT PROCESSING
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=[
        'requests'
    ]
)
def docling_processing_component(
    setup_config: Dict[str, Any],
    document_url: str
) -> NamedTuple("ProcessingOutput", [("processed_content", str)]):
    """
    Process complex academic documents using Docling's advanced document intelligence.
    
    Args:
        setup_config: Configuration from docling_setup_component
        document_url: URL of the document to process (PDF, DOCX, etc.)
        
    Returns:
        NamedTuple containing processed content
    """
    import requests
    from collections import namedtuple
    
    print("Starting Docling Document Intelligence Processing")
    print("=" * 60)
    
    # Extract Docling service configuration
    docling_config = setup_config["document_intelligence"]
    api_address = docling_config["docling_service"]
    timeout = docling_config["processing_timeout"]
    
    # Configure headers (no authentication needed for cluster-internal service)
    headers = {"Content-Type": "application/json"}
    
    print(f"Docling Service: {api_address}/v1alpha/convert/source")
    print(f"Processing document: {document_url}")
    print(f"Timeout configured: {timeout} seconds (for complex document analysis)")
    print(f"Processing may take 1-2 minutes for comprehensive analysis...")
    
    # Configure Docling for maximum intelligence extraction
    payload = {
        "http_sources": [{"url": document_url}],         # Document source URL
        "options": {
            "to_formats": ["md"],                        # Output as structured Markdown
            "image_export_mode": "placeholder"           # Handle images appropriately
        },
    }
    
    try:
        # Submit document for intelligent processing
        print("Submitting document to Docling for intelligent analysis...")
        response = requests.post(
            f"{api_address}/v1alpha/convert/source",
            json=payload,
            headers=headers,
            timeout=timeout  # Extended timeout for complex document analysis
        )
        
        # Verify successful processing
        response.raise_for_status()
        
        # Extract intelligently processed content
        result_data = response.json()
        processed_content = result_data["document"]["md_content"]
        
        
        print("Document Intelligence Processing Complete!")
        print(f"Processed content length: {len(processed_content)} characters")
        print(f"Ready for enhanced RAG ingestion!")
        
        # Content preview for verification
        preview_length = min(500, len(processed_content))
        print(f"\nContent Preview (first {preview_length} characters):")
        print("=" * 60)
        print(processed_content[:preview_length] + ("..." if len(processed_content) > preview_length else ""))
        print("=" * 60)
        
        ProcessingOutput = namedtuple("ProcessingOutput", ["processed_content"])
        return ProcessingOutput(processed_content=processed_content)
        
    except requests.exceptions.Timeout:
        error_msg = f"Document processing timeout after {timeout} seconds"
        print(f"{error_msg}")
        print("Complex documents may require additional processing time")
        raise Exception(error_msg)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Docling processing failed: {e}"
        print(error_msg)
        print("Check Docling service availability and document accessibility")
        raise Exception(error_msg)
        
    except KeyError as e:
        error_msg = f"Unexpected Docling response format: {e}"
        print(error_msg)
        print("Docling service may have returned an unexpected response structure")
        raise Exception(error_msg)

# =============================================================================
# COMPONENT 3: VECTOR DATABASE SETUP
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=['llama_stack_client', 'fire', 'requests']
)
def vector_database_component(
    setup_config: Dict[str, Any]
) -> NamedTuple("VectorDBOutput", [("vector_db_status", Dict[str, Any])]):
    """
    Create and register a vector database optimized for document intelligence RAG operations.
    
    Args:
        setup_config: Configuration from docling_setup_component
        
    Returns:
        NamedTuple containing vector database status and configuration
    """
    from llama_stack_client import LlamaStackClient
    from collections import namedtuple
    
    print("Setting Up Vector Database for Document Intelligence")
    print("=" * 60)
    
    # Extract configuration
    base_url = setup_config["base_url"]
    vector_db_id = setup_config["vector_db_id"]
    doc_intel_config = setup_config["document_intelligence"]
    
    # Initialize LlamaStack client
    print(f"Connecting to LlamaStack: {base_url}")
    client = LlamaStackClient(
        base_url=base_url,
        provider_data=None  # No additional provider configuration needed
    )
    
    print(f"LlamaStack client connected successfully")
    
    # Register vector database for document intelligence
    print(f"Registering vector database for document intelligence...")
    print(f"  • Database ID: {vector_db_id}")
    print(f"  • Embedding Model: {doc_intel_config['embedding_model']}")
    print(f"  • Vector Dimensions: {doc_intel_config['embedding_dimension']}")
    print(f"  • Provider: {doc_intel_config['vector_provider']}")
    
    try:
        # Register the vector database with enhanced configuration for document intelligence
        client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=doc_intel_config["embedding_model"],
            embedding_dimension=doc_intel_config["embedding_dimension"],
            provider_id=doc_intel_config["vector_provider"], # Milvus backend
        )
        
        print("Vector database registered successfully!")
        
        # Prepare status response
        vector_db_status = {
            "status": "success",
            "vector_db_id": vector_db_id,
            "embedding_model": doc_intel_config["embedding_model"],
            "embedding_dimension": doc_intel_config["embedding_dimension"],
            "provider": doc_intel_config["vector_provider"],
            "capabilities": [
                "semantic_search",
                "enhanced_metadata",
                "document_intelligence",
                "complex_chunking"
            ],
            "ready_for_ingestion": True
        }
        
        print("Vector database ready for Docling-processed content ingestion!")
        
        VectorDBOutput = namedtuple("VectorDBOutput", ["vector_db_status"])
        return VectorDBOutput(vector_db_status=vector_db_status)
        
    except Exception as e:
        error_msg = f"Vector database registration failed: {e}"
        print(error_msg)
        print("Check LlamaStack service and Milvus backend availability")
        
        # Return error status
        vector_db_status = {
            "status": "error",
            "error_message": str(e),
            "vector_db_id": vector_db_id,
            "ready_for_ingestion": False
        }
        
        VectorDBOutput = namedtuple("VectorDBOutput", ["vector_db_status"])
        return VectorDBOutput(vector_db_status=vector_db_status)

# =============================================================================
# COMPONENT 4: DOCUMENT INGESTION
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=['llama_stack_client', 'fire', 'requests']
)
def document_ingestion_component(
    setup_config: Dict[str, Any],
    processed_content: str,
    document_url: str,
    vector_db_status: Dict[str, Any]
) -> NamedTuple("IngestionOutput", [("ingestion_results", Dict[str, Any])]):
    """
    Ingest intelligently-processed documents into the RAG system.

    Args:
        setup_config: Configuration from docling_setup_component
        processed_content: Structured Markdown from docling_processing_component
        document_url: URL of the processed document
        vector_db_status: Status from vector_database_component
        
    Returns:
        NamedTuple containing ingestion results
    """
    from llama_stack_client import LlamaStackClient, RAGDocument
    from collections import namedtuple
    
    print("Starting Document Intelligence Ingestion")
    print("=" * 60)
    
    # Verify prerequisites
    if not vector_db_status.get("ready_for_ingestion", False):
        error_msg = "Vector database not ready for ingestion"
        print(error_msg)
        print(f"Vector DB Status: {vector_db_status}")
        raise Exception(error_msg)
    
    # Extract configuration
    base_url = setup_config["base_url"]
    vector_db_id = setup_config["vector_db_id"]
    doc_intel_config = setup_config["document_intelligence"]
    chunk_size = doc_intel_config["chunk_size_tokens"]
    
    print(f"LlamaStack URL: {base_url}")
    print(f"Vector Database ID: {vector_db_id}")
    print(f"Content Length: {len(processed_content)} characters")
    print(f"Chunk Size: {chunk_size} tokens")
    
    # Initialize LlamaStack client
    client = LlamaStackClient(
        base_url=base_url,
        provider_data=None
    )
    
    # Create RAGDocument
    documents = [
        RAGDocument(
            document_id="docling-processed-doc",
            content=processed_content,
            metadata={
                "source_url": document_url,
                "processing_method": "docling",
                "document_type": "academic_paper",
                "has_tables": True,
                "has_formulas": True,
                "has_figures": True,
            },
        )
    ]
    
    print(f"Preparing to ingest intelligently-processed document:")
    print(f"  • Document ID: docling-processed-doc")
    print(f"  • Content length: {len(processed_content)} characters")
    print(f"  • Processing method: Docling document intelligence")
    print(f"  • Content includes: tables, formulas, figures, and structured text")
    
    # Use LlamaStack RAG Tool for intelligent chunking
    try:
        client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=chunk_size,
        )
        
        print("\nDocument ingestion complete!")
        print("Docling-processed content is now searchable via semantic similarity!")
        print("Complex academic content (tables, formulas, figures) is now queryable!")
        
        # Generate ingestion results
        ingestion_results = {
            "status": "success",
            "document_id": "docling-processed-doc",
            "content_length": len(processed_content),
            "chunk_size_tokens": chunk_size,
            "vector_db_id": vector_db_id,
            "ready_for_queries": True,
        }
        
        IngestionOutput = namedtuple("IngestionOutput", ["ingestion_results"])
        return IngestionOutput(ingestion_results=ingestion_results)
        
    except Exception as e:
        error_msg = f"Document ingestion failed: {e}"
        print(error_msg)
        print("Check Docling processing results and vector database configuration")
        
        # Return error results
        ingestion_results = {
            "status": "error",
            "error_message": str(e),
            "document_id": "docling-processed-doc",
            "ready_for_queries": False
        }
        
        IngestionOutput = namedtuple("IngestionOutput", ["ingestion_results"])
        return IngestionOutput(ingestion_results=ingestion_results)

# =============================================================================
# COMPONENT 5: RAG TESTING AND QUERY EXECUTION
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=[
        'llama_stack_client',
        'fire',
        'requests'
    ]
)
def rag_testing_component(
    setup_config: Dict[str, Any],
    ingestion_results: Dict[str, Any],
    test_queries: List[str]
) -> NamedTuple("TestingOutput", [("test_results", List[Dict[str, Any]])]):
    """
    Execute document intelligence queries to test and demonstrate RAG capabilities.
    
    Args:
        setup_config: Configuration from docling_setup_component
        ingestion_results: Results from document_ingestion_component
        test_queries: List of queries to test document intelligence capabilities
        
    Returns:
        NamedTuple containing comprehensive test results for each query
    """
    from llama_stack_client import LlamaStackClient
    from collections import namedtuple
    
    print("Starting Document Intelligence RAG Testing")
    print("=" * 60)
    
    # Verify prerequisites
    if not ingestion_results.get("ready_for_queries", False):
        error_msg = "System not ready for queries"
        print(error_msg)
        print(f"Ingestion Status: {ingestion_results}")
        raise Exception(error_msg)
    
    # Extract configuration
    base_url = setup_config["base_url"]
    vector_db_id = setup_config["vector_db_id"]
    model_config = setup_config["model_config"]
    sampling_params = setup_config["sampling_params"]
    
    print(f"LlamaStack URL: {base_url}")
    print(f"Vector Database ID: {vector_db_id}")
    print(f"Model: {model_config['model_id']}")
    print(f"Test Queries: {len(test_queries)} queries")
    print(f"Testing document intelligence capabilities...")
    
    # Initialize LlamaStack client
    client = LlamaStackClient(
        base_url=base_url,
        provider_data=None
    )
    
    # Execute document intelligence queries
    test_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nExecuting Query {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("-" * 50)
        
        try:
            # STEP 1: RAG Retrieval
            # Use semantic search to find relevant document intelligence chunks
            print("Performing semantic retrieval...")
            rag_response = client.tool_runtime.rag_tool.query(
                content=query,                               # User's question
                vector_db_ids=[vector_db_id],               # Document intelligence database
                query_config={                              # Format retrieved results
                    "chunk_template": "Result {index}\\nContent: {chunk.content}\\nMetadata: {metadata}\\n",
                },
            )
            
            print(f"Retrieved {len(rag_response.metadata.get('chunks', []))} relevant chunks")
            
            # STEP 2: Context Preparation
            # Prepare enhanced prompt with document intelligence context
            messages = [
                {"role": "system", "content": "You are a helpful assistant specializing in document intelligence and academic content analysis."}
            ]
            
            # Inject retrieved document intelligence as context
            prompt_context = rag_response.content
            enhanced_prompt = f"""Please answer the given query using the document intelligence context below.

CONTEXT (Processed with Docling Document Intelligence):
{prompt_context}

QUERY:
{query}

Note: The context includes intelligently processed content with preserved tables, formulas, figures, and document structure."""
            
            messages.append({"role": "user", "content": enhanced_prompt})
            
            # STEP 3: Enhanced Generation
            # Generate response using document intelligence context
            print("Generating response with document intelligence context...")
            response = client.inference.chat_completion(
                messages=messages,
                model_id=model_config["model_id"],
                sampling_params=sampling_params,
                stream=False,  # Simplified for pipeline processing
            )
            
            # Extract response content
            if hasattr(response, 'completion_message'):
                generated_answer = response.completion_message.content
            else:
                generated_answer = str(response)
            
            print(f"Generated response ({len(generated_answer)} characters)")
            
            # STEP 4: Result Analysis
            # Analyze the quality and capabilities demonstrated
            result = {
                "query_id": i,
                "query": query,
                "retrieved_chunks": len(rag_response.metadata.get('chunks', [])),
                "generated_answer": generated_answer,
                "rag_context": prompt_context,
                "rag_metadata": rag_response.metadata,
                "status": "success",
                "demonstrates_capabilities": [
                    "semantic_search",
                    "document_intelligence",
                    "context_aware_generation"
                ]
            }
            
            # Analyze specific document intelligence features demonstrated
            if "table" in query.lower() or "data" in query.lower():
                result["demonstrates_capabilities"].append("table_data_retrieval")
            if "formula" in query.lower() or "equation" in query.lower():
                result["demonstrates_capabilities"].append("mathematical_content")
            if "figure" in query.lower() or "chart" in query.lower():
                result["demonstrates_capabilities"].append("visual_content_understanding")
            
            test_results.append(result)
            
            print(f"Query {i} completed successfully")
            print(f"Demonstrated capabilities: {', '.join(result['demonstrates_capabilities'])}")
            
        except Exception as e:
            error_msg = f"Query {i} failed: {e}"
            print(error_msg)
            
            # Record error result
            error_result = {
                "query_id": i,
                "query": query,
                "status": "error",
                "error_message": str(e),
                "generated_answer": None,
                "demonstrates_capabilities": []
            }
            test_results.append(error_result)
    
    # Summary results
    successful_queries = sum(1 for r in test_results if r["status"] == "success")
    total_capabilities = set()
    for result in test_results:
        total_capabilities.update(result.get("demonstrates_capabilities", []))
    
    print(f"\nDocument Intelligence Testing Complete!")
    print(f"Results Summary:")
    print(f"  • Total Queries: {len(test_queries)}")
    print(f"  • Successful: {successful_queries}")
    print(f"  • Failed: {len(test_queries) - successful_queries}")
    print(f"  • Capabilities Demonstrated: {len(total_capabilities)}")
    print(f"  • Document Intelligence Features: Yes")
    
    if total_capabilities:
        print(f"Demonstrated Capabilities:")
        for capability in sorted(total_capabilities):
            print(f"  • {capability.replace('_', ' ').title()}")
    
    TestingOutput = namedtuple("TestingOutput", ["test_results"])
    return TestingOutput(test_results=test_results)

# =============================================================================
# MAIN PIPELINE DEFINITION
# =============================================================================

@dsl.pipeline(
    name="Document Intelligence RAG Pipeline",
    description="Advanced RAG pipeline with Docling document intelligence for complex academic content processing"
)
def document_intelligence_rag_pipeline(
    document_url: str,
    test_queries: List[str],
    embedding_model: str,
    embedding_dimension: int,
    chunk_size_tokens: int,
    vector_provider: str,
    docling_service: str,
    processing_timeout: int,
    llama_stack_url: str,
    model_id: str,
    temperature: float,
    max_tokens: int
):
    """
    Comprehensive Document Intelligence RAG Pipeline using Docling Processing
    
    Pipeline Stages:
    1. Setup: Initialize LlamaStack client and document intelligence configuration
    2. Processing: Transform documents using Docling's advanced analysis
    3. Storage: Create vector database optimized for document intelligence
    4. Ingestion: Store processed content with enhanced metadata
    5. Testing: Execute queries demonstrating document intelligence capabilities
    
    Args:
        document_url: URL of complex academic document to process
        test_queries: List of queries to test document intelligence (optional)
        embedding_model: Sentence transformer model for text embeddings
        embedding_dimension: Vector dimensions (must match the embedding model)
        chunk_size_tokens: Optimal chunk size for academic content processing
        vector_provider: Vector database backend provider (e.g., "milvus")
        docling_service: URL of the Docling document processing service
        processing_timeout: Timeout in seconds for complex document processing
        llama_stack_url: URL of the LlamaStack service
        model_id: Model identifier for text generation
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens for model responses
        
    Returns:
        Complete pipeline execution with document intelligence capabilities demonstrated
    """
    
    # Default test queries for document intelligence
    if test_queries is None:
        test_queries = [
            "What is the PRFXception mentioned in the document?",
            "Can you provide the accuracy values of overall model prediction and residual cross-validation for five regions in southeast Tibet and four regions in northwest Yunnan?",
            "What tables are present in this document and what data do they contain?",
            "Are there any mathematical formulas or equations in the document? What do they represent?",
            "What is the structure and organization of this academic paper?"
        ]
    
    
    
    # STAGE 1: Document Intelligence Setup
    setup_task = docling_setup_component(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        chunk_size_tokens=chunk_size_tokens,
        vector_provider=vector_provider,
        docling_service=docling_service,
        processing_timeout=processing_timeout,
        llama_stack_url=llama_stack_url,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # STAGE 2: Docling Document Processing
    processing_task = docling_processing_component(
        setup_config=setup_task.outputs["setup_config"],
        document_url=document_url
    )
    processing_task.after(setup_task)
    
    # STAGE 3: Vector Database Creation
    vector_db_task = vector_database_component(
        setup_config=setup_task.outputs["setup_config"]
    )
    vector_db_task.after(setup_task)
    
    # STAGE 4: Document Ingestion
    ingestion_task = document_ingestion_component(
        setup_config=setup_task.outputs["setup_config"],
        processed_content=processing_task.outputs["processed_content"],
        document_url=document_url,
        vector_db_status=vector_db_task.outputs["vector_db_status"]
    )
    ingestion_task.after(processing_task)
    ingestion_task.after(vector_db_task)
    
    # STAGE 5: RAG Testing
    testing_task = rag_testing_component(
        setup_config=setup_task.outputs["setup_config"],
        ingestion_results=ingestion_task.outputs["ingestion_results"],
        test_queries=test_queries
    )
    testing_task.after(ingestion_task)
    

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

if __name__ == '__main__':
    """
    Execute the Document Intelligence RAG Pipeline
    """
    
    # === Pipeline Configuration ===
    # Configure the pipeline with document intelligence optimized parameters
    arguments = {
        "document_url": "https://arxiv.org/pdf/2404.14661",
        "test_queries": [
            "What is the PRFXception mentioned in the document?",
        ],
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "chunk_size_tokens": 512,
        "vector_provider": "milvus",
        "docling_service": "http://docling-v0-7-0-predictor.ai501.svc.cluster.local:5001",
        "processing_timeout": 180,
        "llama_stack_url": "http://llama-stack-service:8321",
        "model_id": "llama32",
        "temperature": 0.0,
        "max_tokens": 4096
    }
    
    # === Kubernetes Configuration ===
    # Get namespace and configure Kubeflow connection
    namespace_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint = f'https://ds-pipeline-dspa.{namespace}.svc:8443'

    # Configure authentication
    sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(sa_token_file_path, 'r') as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert = '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    
    # Create Kubeflow client and execute pipeline
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert
    )

    # Execute the document intelligence pipeline
    client.create_run_from_pipeline_func(
        document_intelligence_rag_pipeline,
        arguments=arguments,
        experiment_name="document-intelligence-rag",
        enable_caching=False  # Disable caching for fresh document intelligence processing
    )
    
    print("=" * 60)
    print("📄 DOCUMENT INTELLIGENCE RAG PIPELINE SUBMITTED")
    print("=" * 60)
    print(f"🔗 Document URL: {arguments['document_url']}")
    print(f"🧪 Experiment: document-intelligence-rag")
    print(f"🤖 Model: all-MiniLM-L6-v2 (384D)")
    print(f"⚙️  Chunk Size: {arguments['chunk_size_tokens']} tokens")
    print(f"📊 Vector DB: {arguments['vector_provider']}")
    print(f"🔬 Docling Service: Active")
    print(f"❓ Test Queries: {len(arguments['test_queries'])}")
    print("=" * 60)
    print("Pipeline will execute 5 stages: Setup → Processing → Vector DB → Ingestion → Testing")
    print("Monitor progress in the Kubeflow UI")