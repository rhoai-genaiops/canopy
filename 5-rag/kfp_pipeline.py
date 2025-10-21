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
from kfp import dsl, components, kubernetes
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
        'fire',
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
    max_tokens: int,
    vector_db_id: str,
    vector_db_alias: Optional[str] = None
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
        vector_db_id: Vector database identifier (used by Canopy backend for queries)
        vector_db_alias: Optional alias for the vector database (e.g., "latest")

    Returns:
        NamedTuple containing setup configuration for downstream components
        NamedTuple usage: https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/parameters/#multiple-output-parameters
    """
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
        "vector_db_id": vector_db_id,  # Configurable vector database identifier
        "vector_db_alias": vector_db_alias  # Optional alias (e.g., "latest")
    }
    
    print(f"Document Intelligence Setup Complete:")
    print(f"  - LlamaStack URL: {base_url}")
    print(f"  - Model: {model_config['model_id']}")
    print(f"  - Strategy: {sampling_strategy['type']}")
    print(f"  - Max Tokens: {model_config['max_tokens']}")
    print(f"  - Embedding Model: {document_intelligence_config['embedding_model']}")
    print(f"  - Vector Database ID: {setup_config['vector_db_id']}")
    if vector_db_alias:
        print(f"  - Vector Database Alias: {vector_db_alias}")
    print(f"  - Docling Service: {document_intelligence_config['docling_service']}")
    print("Ready for intelligent document processing!")
    
    # Return configuration for downstream components
    SetupOutput = namedtuple("SetupOutput", ["setup_config"])
    return SetupOutput(setup_config=setup_config)

# =============================================================================
# COMPONENT 2: MINIO LIST AND DOWNLOAD ALL DOCUMENTS
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=[
        'boto3==1.34.103'
    ]
)
def download_all_from_minio_component(
    bucket_name: str
) -> NamedTuple("MinIODownloadAllOutput", [("downloaded_files", List[str]), ("original_keys", List[str]), ("file_count", int)]):
    """
    List and download all documents from a MinIO bucket.

    Args:
        bucket_name: Name of the bucket containing documents

    Returns:
        NamedTuple containing:
        - downloaded_files: List of local file paths
        - original_keys: List of original MinIO object keys (preserves special characters)
        - file_count: Number of successfully downloaded files

    Environment Variables (from Kubernetes secret):
        AWS_S3_ENDPOINT: MinIO server endpoint URL
        AWS_ACCESS_KEY_ID: MinIO access key
        AWS_SECRET_ACCESS_KEY: MinIO secret key
    """
    import boto3
    from collections import namedtuple
    import os

    print("Listing and Downloading All Documents from MinIO")
    print("=" * 60)

    # Read MinIO credentials from environment variables
    minio_endpoint = os.environ.get('AWS_S3_ENDPOINT')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    if not all([minio_endpoint, access_key, secret_key]):
        raise Exception("Missing MinIO credentials in environment variables. Expected: AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")

    print(f"MinIO Endpoint: {minio_endpoint}")
    print(f"Bucket: {bucket_name}")

    # Create shared volume directory if it doesn't exist
    shared_volume_path = "/shared-data/documents"
    os.makedirs(shared_volume_path, exist_ok=True)

    try:
        # Initialize MinIO client using boto3
        s3_client = boto3.client(
            's3',
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            verify=False  # For self-signed certificates
        )

        print(f"Listing objects in bucket '{bucket_name}'...")

        # List all objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)

        if 'Contents' not in response or len(response['Contents']) == 0:
            print(f"No objects found in bucket '{bucket_name}'")
            MinIODownloadAllOutput = namedtuple("MinIODownloadAllOutput", ["downloaded_files", "original_keys", "file_count"])
            return MinIODownloadAllOutput(downloaded_files=[], original_keys=[], file_count=0)

        objects = response['Contents']
        print(f"Found {len(objects)} objects in bucket")

        downloaded_files = []
        original_keys = []

        # Download each object
        for obj in objects:
            object_key = obj['Key']

            # Skip directories (objects ending with /)
            if object_key.endswith('/'):
                print(f"Skipping directory: {object_key}")
                continue

            # Create local file path preserving the directory structure
            # Replace / with _ to flatten the structure
            # Keep the original filename to preserve special characters
            # This works fine with spaces, parentheses, unicode characters, etc.
            safe_filename = object_key.replace('/', '_')
            local_file_path = f"{shared_volume_path}/{safe_filename}"

            print(f"Downloading: {object_key}")
            print(f"  -> Local path: {local_file_path}")

            try:
                # boto3 download_file handles special characters in object_key automatically
                s3_client.download_file(bucket_name, object_key, local_file_path)
                file_size = os.path.getsize(local_file_path)
                print(f"  [OK] Downloaded {file_size} bytes")

                # Store both the local path and original object key for metadata tracking
                downloaded_files.append(local_file_path)
                original_keys.append(object_key)  # Preserve original key with all special characters
            except Exception as e:
                print(f"  [FAILED] Failed to download {object_key}: {e}")
                # Continue with other files even if one fails
                continue

        print("=" * 60)
        print(f"Download Summary:")
        print(f"  - Total objects in bucket: {len(objects)}")
        print(f"  - Successfully downloaded: {len(downloaded_files)}")
        print(f"  - Files:")
        for orig_key, file_path in zip(original_keys, downloaded_files):
            print(f"    - {orig_key} -> {file_path}")
        print("=" * 60)

        MinIODownloadAllOutput = namedtuple("MinIODownloadAllOutput", ["downloaded_files", "original_keys", "file_count"])
        return MinIODownloadAllOutput(
            downloaded_files=downloaded_files,
            original_keys=original_keys,
            file_count=len(downloaded_files)
        )

    except Exception as e:
        error_msg = f"Failed to list/download documents from MinIO: {e}"
        print(error_msg)
        print("Check MinIO credentials and bucket name")
        raise Exception(error_msg)

# =============================================================================
# COMPONENT 3: BATCH DOCLING DOCUMENT PROCESSING
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=[
        'requests'
    ]
)
def batch_docling_processing_component(
    setup_config: Dict[str, Any],
    downloaded_files: List[str],
    original_keys: List[str],
    file_count: int
) -> NamedTuple("BatchProcessingOutput", [("processed_files", List[str]), ("original_keys", List[str]), ("processed_count", int)]):
    """
    Process multiple academic documents using Docling's advanced document intelligence.

    Args:
        setup_config: Configuration from docling_setup_component
        downloaded_files: List of local file paths to process
        original_keys: List of original MinIO object keys (with special characters preserved)
        file_count: Number of files to process

    Returns:
        NamedTuple containing list of processed content file paths, original keys, and count
    """
    import requests
    from collections import namedtuple
    import os
    import uuid

    print("Starting Batch Docling Document Intelligence Processing")
    print("=" * 60)

    # Extract Docling service configuration
    docling_config = setup_config["document_intelligence"]
    api_address = docling_config["docling_service"]
    timeout = docling_config["processing_timeout"]

    print(f"Docling Service: {api_address}/v1alpha/convert/file")
    print(f"Processing {file_count} documents")
    print(f"Timeout configured: {timeout} seconds per document")
    print(f"Total estimated time: ~{file_count * 2} minutes")
    print("=" * 60)

    if file_count == 0 or not downloaded_files:
        print("No files to process")
        BatchProcessingOutput = namedtuple("BatchProcessingOutput", ["processed_files", "original_keys", "processed_count"])
        return BatchProcessingOutput(processed_files=[], original_keys=[], processed_count=0)

    # Create shared volume directory for processed content
    shared_volume_path = "/shared-data/processed"
    os.makedirs(shared_volume_path, exist_ok=True)

    processed_files = []
    processed_original_keys = []  # Track original keys only for successfully processed files

    # Process each document (original_keys passed through for metadata tracking)
    for idx, document_path in enumerate(downloaded_files, 1):
        print(f"\nProcessing document {idx}/{file_count}: {document_path}")
        print("-" * 60)

        # Verify file exists
        if not os.path.exists(document_path):
            print(f"  [FAILED] File not found: {document_path}")
            continue

        # Read the document file
        try:
            with open(document_path, 'rb') as f:
                file_content = f.read()
            print(f"  - Read {len(file_content)} bytes")
        except Exception as e:
            print(f"  [FAILED] Failed to read file: {e}")
            continue

        try:
            # ---- begin patched upload logic for v1alpha/convert/file ----
            import json, unicodedata, re

            # Derive a safe ASCII filename if we have one; else fall back
            orig_name = os.path.basename(document_path) if document_path else ""
            def _safe_ascii_name(name: str, default="upload.pdf"):
                if not name:
                    return default
                # normalize & strip non-ascii
                s = unicodedata.normalize("NFKD", name)
                s = s.encode("ascii", "ignore").decode("ascii")
                # keep only simple filename chars
                s = re.sub(r"[^A-Za-z0-9._-]", "_", s).strip("._")
                if not s:
                    s = default
                if not s.lower().endswith(".pdf"):
                    s += ".pdf"
                return s

            safe_name = _safe_ascii_name(orig_name, "upload.pdf")

            print(f"  - Submitting to Docling...")

            # Docling alpha prefers 'files' (plural) for multipart uploads
            files = {
                "files": (safe_name, file_content, "application/pdf")
            }

            # Try two encodings for options; some alpha builds expect flat fields,
            # others expect an 'options' JSON blob.
            attempts = [
                {
                    "desc": "Attempt A: flat fields",
                    "data": {
                        "to_formats": "md",                 # server will coerce to list
                        "image_export_mode": "placeholder",
                    },
                },
                {
                    "desc": "Attempt B: options JSON",
                    "data": {
                        "options": json.dumps({
                            "to_formats": ["md"],
                            "image_export_mode": "placeholder",
                        })
                    },
                },
            ]

            last_err_text = None
            for attempt in attempts:
                print(f"    -> {attempt['desc']}")
                response = requests.post(
                    f"{api_address}/v1alpha/convert/file",
                    files=files,
                    data=attempt["data"],
                    timeout=timeout,
                )
                if response.status_code == 422:
                    # validation mismatch; try next encoding
                    print(f"       got 422, retrying with alternate encoding")
                    last_err_text = f"422 {response.text[:800]}"
                    continue
                if response.status_code >= 400:
                    # other error; break and raise
                    last_err_text = f"{response.status_code} {response.text[:800]}"
                    response.raise_for_status()
                # success
                break
            else:
                raise requests.exceptions.RequestException(
                    f"Docling upload failed after retries. Last error: {last_err_text}"
                )

            result_data = response.json()

            def _extract_md_payload(payload):
                # Handle multiple shapes returned by different alpha builds
                if isinstance(payload, dict):
                    doc = payload.get("document")
                    if isinstance(doc, dict):
                        if isinstance(doc.get("md_content"), str):
                            return doc["md_content"]
                        if isinstance(doc.get("md"), str):
                            return doc["md"]
                        if isinstance(doc.get("content"), dict) and isinstance(doc["content"].get("md"), str):
                            return doc["content"]["md"]
                    docs = payload.get("documents")
                    if isinstance(docs, list) and docs:
                        first = docs[0]
                        if isinstance(first, dict):
                            if isinstance(first.get("md_content"), str):
                                return first["md_content"]
                            if isinstance(first.get("md"), str):
                                return first["md"]
                    res = payload.get("result")
                    if isinstance(res, dict) and isinstance(res.get("md"), str):
                        return res["md"]
                raise KeyError(f"Could not find markdown in response. Top-level keys: {list(payload) if isinstance(payload, dict) else type(payload).__name__}")

            processed_content = _extract_md_payload(result_data)
            print(f"  [OK] Processed {len(processed_content)} characters")
            # ---- end patched upload logic ----

            # Write content to shared volume
            unique_id = str(uuid.uuid4())
            content_file_path = f"{shared_volume_path}/processed_{idx}_{unique_id}.txt"

            with open(content_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)

            print(f"  [OK] Saved to: {content_file_path}")
            processed_files.append(content_file_path)
            # Only add original key if processing succeeded
            processed_original_keys.append(original_keys[idx - 1])

        except requests.exceptions.Timeout:
            print(f"  [FAILED] Timeout after {timeout} seconds")
            continue

        except requests.exceptions.RequestException as e:
            print(f"  [FAILED] Docling processing failed: {e}")
            continue

        except KeyError as e:
            print(f"  [FAILED] Unexpected response format: {e}")
            continue

        except Exception as e:
            print(f"  [FAILED] Unexpected error: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Batch Processing Summary:")
    print(f"  - Total documents: {file_count}")
    print(f"  - Successfully processed: {len(processed_files)}")
    print(f"  - Failed: {file_count - len(processed_files)}")
    print("=" * 60)

    # Return only the keys for successfully processed files
    # This ensures processed_files and processed_original_keys have matching indices
    BatchProcessingOutput = namedtuple("BatchProcessingOutput", ["processed_files", "original_keys", "processed_count"])
    return BatchProcessingOutput(
        processed_files=processed_files,
        original_keys=processed_original_keys,  # Only keys for successfully processed files
        processed_count=len(processed_files)
    )

# =============================================================================
# COMPONENT 4: VECTOR DATABASE SETUP
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=['llama_stack_client', 'fire', 'requests']
)
def vector_database_component(
    setup_config: Dict[str, Any]
) -> NamedTuple("VectorDBOutput", [("vector_db_status", Dict[str, Any]), ("vector_db_ids", List[str])]):
    """
    Create and register vector database(s) optimized for document intelligence RAG operations.
    If an alias is provided, creates two separate vector databases for redundancy.

    Args:
        setup_config: Configuration from docling_setup_component

    Returns:
        NamedTuple containing:
        - vector_db_status: Status information for the main database
        - vector_db_ids: List of all created database IDs (for ingestion)
    """
    from llama_stack_client import LlamaStackClient
    from collections import namedtuple
    
    print("Setting Up Vector Database for Document Intelligence")
    print("=" * 60)
    
    # Extract configuration
    base_url = setup_config["base_url"]
    vector_db_id = setup_config["vector_db_id"]
    vector_db_alias = setup_config.get("vector_db_alias")
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
    print(f"  - Database ID: {vector_db_id}")
    print(f"  - Embedding Model: {doc_intel_config['embedding_model']}")
    print(f"  - Vector Dimensions: {doc_intel_config['embedding_dimension']}")
    print(f"  - Provider: {doc_intel_config['vector_provider']}")
    if vector_db_alias:
        print(f"  - Alias: {vector_db_alias}")

    try:
        # Check if vector database already exists
        print(f"Checking if vector database '{vector_db_id}' already exists...")
        try:
            existing_dbs = client.vector_dbs.list()
            print(f"Existing vector databases: {existing_dbs}")
            # If the DB already exists, we might get a conflict error on registration
            # In that case, we should treat it as success
        except Exception as list_error:
            print(f"Warning: Could not list existing vector databases: {list_error}")

        # Register the vector database with enhanced configuration for document intelligence
        print(f"Attempting to register vector database '{vector_db_id}'...")
        client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=doc_intel_config["embedding_model"],
            embedding_dimension=doc_intel_config["embedding_dimension"],
            provider_id=doc_intel_config["vector_provider"], # Milvus backend
        )

        print("Vector database registered successfully!")

        # Track all created database IDs
        created_db_ids = [vector_db_id]

        # Create additional vector database with alias name if provided
        # This creates a completely separate database (not an alias)
        if vector_db_alias:
            print(f"\nCreating additional vector database '{vector_db_alias}'...")
            print(f"Note: This creates a separate database, not an alias")
            try:
                client.vector_dbs.register(
                    vector_db_id=vector_db_alias,
                    embedding_model=doc_intel_config["embedding_model"],
                    embedding_dimension=doc_intel_config["embedding_dimension"],
                    provider_id=doc_intel_config["vector_provider"], # Milvus backend
                )
                print(f"Additional database '{vector_db_alias}' registered successfully!")
                created_db_ids.append(vector_db_alias)
                print(f"Documents will be ingested into BOTH databases")
            except Exception as alias_error:
                print(f"Warning: Failed to register additional database '{vector_db_alias}': {alias_error}")
                print("Continuing with main vector database only...")

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
            "ready_for_ingestion": True,
            "all_db_ids": created_db_ids  # List of all created databases
        }

        print(f"\nVector database setup complete!")
        print(f"Created {len(created_db_ids)} database(s): {created_db_ids}")
        print(f"Ready for Docling-processed content ingestion!")

        VectorDBOutput = namedtuple("VectorDBOutput", ["vector_db_status", "vector_db_ids"])
        return VectorDBOutput(vector_db_status=vector_db_status, vector_db_ids=created_db_ids)
        
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

        VectorDBOutput = namedtuple("VectorDBOutput", ["vector_db_status", "vector_db_ids"])
        return VectorDBOutput(vector_db_status=vector_db_status, vector_db_ids=[])

# =============================================================================
# COMPONENT 5: BATCH DOCUMENT INGESTION
# =============================================================================

@component(
    base_image='python:3.11',
    packages_to_install=['llama_stack_client', 'fire', 'requests']
)
def batch_document_ingestion_component(
    setup_config: Dict[str, Any],
    processed_files: List[str],
    original_keys: List[str],
    processed_count: int,
    bucket_name: str,
    vector_db_status: Dict[str, Any],
    vector_db_ids: List[str]
) -> NamedTuple("BatchIngestionOutput", [("ingestion_results", Dict[str, Any])]):
    """
    Ingest multiple intelligently-processed documents into the RAG system.
    If multiple vector database IDs are provided, ingests into all of them.

    Args:
        setup_config: Configuration from docling_setup_component
        processed_files: List of paths to files containing processed content
        original_keys: List of original MinIO object keys (preserves special characters like "Syllabus (4-year).pdf")
        processed_count: Number of processed files
        bucket_name: MinIO bucket name (used for source identification)
        vector_db_status: Status from vector_database_component
        vector_db_ids: List of vector database IDs to ingest into

    Returns:
        NamedTuple containing batch ingestion results
    """
    from llama_stack_client import LlamaStackClient, RAGDocument
    from collections import namedtuple
    import os

    print("Starting Batch Document Intelligence Ingestion")
    print("=" * 60)

    # Verify prerequisites
    if not vector_db_status.get("ready_for_ingestion", False):
        error_msg = "Vector database not ready for ingestion"
        print(error_msg)
        print(f"Vector DB Status: {vector_db_status}")
        raise Exception(error_msg)

    if processed_count == 0 or not processed_files:
        print("No processed files to ingest")
        BatchIngestionOutput = namedtuple("BatchIngestionOutput", ["ingestion_results"])
        return BatchIngestionOutput(ingestion_results={
            "status": "success",
            "documents_ingested": 0,
            "ready_for_queries": True
        })

    # Extract configuration
    base_url = setup_config["base_url"]
    doc_intel_config = setup_config["document_intelligence"]
    chunk_size = doc_intel_config["chunk_size_tokens"]

    # Use provided vector_db_ids list, or fallback to main DB ID from config
    if not vector_db_ids or len(vector_db_ids) == 0:
        vector_db_ids = [setup_config["vector_db_id"]]

    print(f"LlamaStack URL: {base_url}")
    print(f"Vector Database IDs: {vector_db_ids}")
    print(f"Documents to ingest: {processed_count}")
    print(f"Chunk Size: {chunk_size} tokens")
    print(f"Will ingest into {len(vector_db_ids)} database(s)")
    print("=" * 60)

    # Initialize LlamaStack client
    client = LlamaStackClient(
        base_url=base_url,
        provider_data=None
    )

    documents_to_ingest = []
    successful_reads = 0

    # Read and prepare all processed files
    for idx, content_file_path in enumerate(processed_files, 1):
        print(f"\nReading processed file {idx}/{processed_count}: {content_file_path}")

        try:
            with open(content_file_path, 'r', encoding='utf-8') as f:
                processed_content = f.read()

            print(f"  [OK] Read {len(processed_content)} characters")

            # Use original MinIO key for accurate source tracking
            # This preserves special characters like spaces, parentheses, en-dashes, etc.
            original_key = original_keys[idx - 1] if idx <= len(original_keys) else f"unknown_{idx}"
            doc_id = f"doc_{idx}"

            print(f"  - Original source: {original_key}")

            # Create RAGDocument with original MinIO path
            documents_to_ingest.append(
                RAGDocument(
                    document_id=doc_id,
                    content=processed_content,
                    metadata={
                        "source": f"minio://{bucket_name}/{original_key}",  # Use original key with special characters
                        "original_filename": original_key,  # Preserve filename with special characters
                        "processing_method": "docling",
                        "document_type": "academic_paper",
                        "has_tables": True,
                        "has_formulas": True,
                        "has_figures": True,
                        "batch_index": idx
                    },
                )
            )
            successful_reads += 1

        except FileNotFoundError:
            print(f"  [FAILED] File not found: {content_file_path}")
            continue
        except Exception as e:
            print(f"  [FAILED] Error reading file: {e}")
            continue

    if successful_reads == 0:
        error_msg = "No documents could be read for ingestion"
        print(error_msg)
        BatchIngestionOutput = namedtuple("BatchIngestionOutput", ["ingestion_results"])
        return BatchIngestionOutput(ingestion_results={
            "status": "error",
            "error_message": error_msg,
            "documents_ingested": 0,
            "ready_for_queries": False
        })

    print("\n" + "=" * 60)
    print(f"Ingesting {successful_reads} documents into {len(vector_db_ids)} vector database(s)...")
    print(f"  - Chunk Size: {chunk_size} tokens")

    # Ingest all documents into each vector database
    ingestion_errors = []
    successfully_ingested_dbs = []

    for vector_db_id in vector_db_ids:
        print(f"\n--- Ingesting into database: {vector_db_id} ---")
        try:
            client.tool_runtime.rag_tool.insert(
                documents=documents_to_ingest,
                vector_db_id=vector_db_id,
                chunk_size_in_tokens=chunk_size,
            )
            print(f"[OK] Successfully ingested {successful_reads} documents into '{vector_db_id}'")
            successfully_ingested_dbs.append(vector_db_id)

        except Exception as e:
            error_msg = f"Failed to ingest into '{vector_db_id}': {e}"
            print(f"[FAILED] {error_msg}")
            ingestion_errors.append({"vector_db_id": vector_db_id, "error": str(e)})

    print("\n" + "=" * 60)
    print(f"Batch document ingestion complete!")
    print(f"  - Successfully ingested: {successful_reads} documents")
    print(f"  - Databases ingested into: {len(successfully_ingested_dbs)}/{len(vector_db_ids)}")
    if successfully_ingested_dbs:
        print(f"  - Successful databases: {successfully_ingested_dbs}")
    if ingestion_errors:
        print(f"  - Failed databases: {[err['vector_db_id'] for err in ingestion_errors]}")
    print("=" * 60)

    # Determine overall status
    if len(successfully_ingested_dbs) > 0:
        # At least one database succeeded
        ingestion_results = {
            "status": "success",
            "documents_ingested": successful_reads,
            "total_processed": processed_count,
            "chunk_size_tokens": chunk_size,
            "vector_db_ids": successfully_ingested_dbs,
            "failed_db_ids": [err['vector_db_id'] for err in ingestion_errors],
            "ready_for_queries": True,
        }
    else:
        # All databases failed
        ingestion_results = {
            "status": "error",
            "error_message": f"Failed to ingest into all databases. Errors: {ingestion_errors}",
            "documents_ingested": 0,
            "ready_for_queries": False
        }

    BatchIngestionOutput = namedtuple("BatchIngestionOutput", ["ingestion_results"])
    return BatchIngestionOutput(ingestion_results=ingestion_results)

# =============================================================================
# COMPONENT 6: PIPELINE COMPLETION
# =============================================================================

@component(
    base_image='python:3.11'
)
def pipeline_completion_component(
    test_ingestion_results: Dict[str, Any],
    prod_ingestion_results: Dict[str, Any]
) -> NamedTuple("CompletionOutput", [("completion_status", str)]):
    """
    Final convergence point for the pipeline.
    Ensures the pipeline has a single terminal node for proper completion detection.

    Args:
        test_ingestion_results: Results from test environment ingestion
        prod_ingestion_results: Results from prod environment ingestion

    Returns:
        NamedTuple with completion status
    """
    from collections import namedtuple

    print("Pipeline Completion Check")
    print("=" * 60)

    test_status = test_ingestion_results.get("status", "unknown")
    prod_status = prod_ingestion_results.get("status", "unknown")

    print(f"Test Environment Ingestion: {test_status}")
    print(f"  - Documents Ingested: {test_ingestion_results.get('documents_ingested', 0)}")
    print(f"  - Vector DB IDs: {test_ingestion_results.get('vector_db_ids', [])}")

    print(f"\nProd Environment Ingestion: {prod_status}")
    print(f"  - Documents Ingested: {prod_ingestion_results.get('documents_ingested', 0)}")
    print(f"  - Vector DB IDs: {prod_ingestion_results.get('vector_db_ids', [])}")

    if test_status == "success" and prod_status == "success":
        completion_status = "SUCCESS"
        print(f"\n{completion_status}: Both test and prod ingestion completed successfully!")
    elif test_status == "success" or prod_status == "success":
        completion_status = "PARTIAL_SUCCESS"
        print(f"\n{completion_status}: At least one environment succeeded")
    else:
        completion_status = "FAILED"
        print(f"\n{completion_status}: Both environments failed")

    print("=" * 60)

    CompletionOutput = namedtuple("CompletionOutput", ["completion_status"])
    return CompletionOutput(completion_status=completion_status)

# =============================================================================
# MAIN PIPELINE DEFINITION
# =============================================================================

@dsl.pipeline(
    name="Document Intelligence RAG Pipeline",
    description="Advanced RAG pipeline with Docling document intelligence for complex academic content processing"
)
def document_intelligence_rag_pipeline(
    minio_secret_name: str,
    minio_bucket_name: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_size_tokens: int,
    vector_provider: str,
    docling_service: str,
    processing_timeout: int,
    llama_stack_url: str,
    prod_llama_stack_url: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    vector_db_id: str,
    test_vector_db_alias: Optional[str] = None
):
    """
    Comprehensive Batch Document Intelligence RAG Pipeline with MinIO Integration

    Pipeline Stages:
    1. Setup: Initialize LlamaStack client and document intelligence configuration (Test & Prod)
    2. Batch Download: Download ALL documents from MinIO bucket using credentials from secret
    3. Batch Processing: Process all downloaded documents using Docling's advanced analysis
    4. Vector Database: Create and register vector databases (Test & Prod)
    5. Batch Ingestion: Ingest ALL processed documents with enhanced metadata (Test & Prod)

    MinIO Batch Integration:
    - ALL documents in the bucket are automatically downloaded
    - No need to specify individual file names
    - MinIO credentials (endpoint, access_key, secret_key) are retrieved from Kubernetes secret
    - Secret name is configurable via pipeline parameter
    - Credentials are injected as environment variables into the download component

    Batch Processing Features:
    - Downloads all files from specified MinIO bucket
    - Processes each document through Docling in sequence
    - Ingests all processed documents into vector database
    - Handles failures gracefully - continues with remaining documents
    - Provides comprehensive summary of processing results

    Args:
        minio_secret_name: Name of Kubernetes secret containing MinIO credentials
                          Secret must have keys: AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        minio_bucket_name: MinIO bucket name containing ALL documents to process
        embedding_model: Sentence transformer model for text embeddings
        embedding_dimension: Vector dimensions (must match the embedding model)
        chunk_size_tokens: Optimal chunk size for academic content processing
        vector_provider: Vector database backend provider (e.g., "milvus")
        docling_service: URL of the Docling document processing service
        processing_timeout: Timeout in seconds per document for processing
        llama_stack_url: URL of the test LlamaStack service
        prod_llama_stack_url: URL of the production LlamaStack service
        model_id: Model identifier for text generation
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens for model responses
        vector_db_id: Vector database identifier (used by Canopy backend for queries)
        test_vector_db_alias: Optional alias for the test vector database (e.g., "latest")

    Returns:
        Complete pipeline execution with batch document intelligence capabilities demonstrated
    """
    
    # Use existing PVC for content transfer
    pvc_name = "canopy-workspace-pvc"

    # STAGE 1: Document Intelligence Setup - Test Environment
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
        max_tokens=max_tokens,
        vector_db_id=vector_db_id,
        vector_db_alias=test_vector_db_alias
    )

    # STAGE 1b: Document Intelligence Setup - Prod Environment
    setup_task_prod = docling_setup_component(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        chunk_size_tokens=chunk_size_tokens,
        vector_provider=vector_provider,
        docling_service=docling_service,
        processing_timeout=processing_timeout,
        llama_stack_url=prod_llama_stack_url,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        vector_db_id=vector_db_id
    )

    # STAGE 2: Batch Download - Download ALL documents from MinIO bucket
    download_task = download_all_from_minio_component(
        bucket_name=minio_bucket_name
    )
    # Mount PVC for storing downloaded files
    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc_name,
        mount_path='/shared-data',
    )
    # Inject MinIO credentials from Kubernetes secret as environment variables
    kubernetes.use_secret_as_env(
        download_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
        },
    )

    # STAGE 3: Batch Docling Processing - Process ALL downloaded documents
    processing_task = batch_docling_processing_component(
        setup_config=setup_task.outputs["setup_config"],
        downloaded_files=download_task.outputs["downloaded_files"],
        original_keys=download_task.outputs["original_keys"],
        file_count=download_task.outputs["file_count"]
    )
    # Dependencies automatically handled by output references
    # Mount PVC for content transfer
    kubernetes.mount_pvc(
        processing_task,
        pvc_name=pvc_name,
        mount_path='/shared-data',
    )
    
    # STAGE 4: Vector Database Creation - Test Environment
    vector_db_task = vector_database_component(
        setup_config=setup_task.outputs["setup_config"]
    )
    # Dependencies automatically handled by output references

    # STAGE 4b: Vector Database Creation - Prod Environment
    vector_db_task_prod = vector_database_component(
        setup_config=setup_task_prod.outputs["setup_config"]
    )
    # Dependencies automatically handled by output references

    # STAGE 5: Batch Document Ingestion - Test Environment
    # Test environment ingests into BOTH databases if alias is provided
    ingestion_task = batch_document_ingestion_component(
        setup_config=setup_task.outputs["setup_config"],
        processed_files=processing_task.outputs["processed_files"],
        original_keys=processing_task.outputs["original_keys"],
        processed_count=processing_task.outputs["processed_count"],
        bucket_name=minio_bucket_name,
        vector_db_status=vector_db_task.outputs["vector_db_status"],
        vector_db_ids=vector_db_task.outputs["vector_db_ids"]
    )
    # Dependencies automatically handled by output references
    # Mount same PVC for content access
    kubernetes.mount_pvc(
        ingestion_task,
        pvc_name=pvc_name,
        mount_path='/shared-data',
    )

    # STAGE 5b: Batch Document Ingestion - Prod Environment
    # Prod environment ingests into single database only
    ingestion_task_prod = batch_document_ingestion_component(
        setup_config=setup_task_prod.outputs["setup_config"],
        processed_files=processing_task.outputs["processed_files"],
        original_keys=processing_task.outputs["original_keys"],
        processed_count=processing_task.outputs["processed_count"],
        bucket_name=minio_bucket_name,
        vector_db_status=vector_db_task_prod.outputs["vector_db_status"],
        vector_db_ids=vector_db_task_prod.outputs["vector_db_ids"]
    )
    # Dependencies automatically handled by output references
    # Mount same PVC for content access
    kubernetes.mount_pvc(
        ingestion_task_prod,
        pvc_name=pvc_name,
        mount_path='/shared-data',
    )

    # STAGE 6: Pipeline Completion
    # Creates a single terminal node to ensure proper pipeline completion
    completion_task = pipeline_completion_component(
        test_ingestion_results=ingestion_task.outputs["ingestion_results"],
        prod_ingestion_results=ingestion_task_prod.outputs["ingestion_results"]
    )
    # Dependencies automatically handled by output references

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
        "minio_secret_name": "documents", 
        "minio_bucket_name": "documents",  
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "chunk_size_tokens": 512,
        "vector_provider": "milvus",
        "docling_service": "http://docling-v0-7-0-predictor.ai501.svc.cluster.local:5001",
        "processing_timeout": 180,
        "llama_stack_url": "http://llama-stack-service:8321",
        "prod_llama_stack_url": "http://llama-stack-service-prod:8321",  # Replace with actual prod endpoint
        "model_id": "llama32",
        "temperature": 0.0,
        "max_tokens": 4096,
        "vector_db_id": "docling_vector_db_genaiops",  # Vector database identifier for Canopy backend
        "test_vector_db_alias": "latest"  # Alias for test environment vector database
    }

    COMPILE = True

    if COMPILE:
        kfp.compiler.Compiler().compile(document_intelligence_rag_pipeline, 'document-intelligence-rag.yaml', pipeline_parameters=arguments)
    else:        
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
        print("BATCH DOCUMENT INTELLIGENCE RAG PIPELINE SUBMITTED")
        print("=" * 60)
        print(f"MinIO Bucket: {arguments['minio_bucket_name']} (ALL files will be processed)")
        print(f"Secret Name: {arguments['minio_secret_name']}")
        print(f"Experiment: document-intelligence-rag")
        print(f"Model: all-MiniLM-L6-v2 (384D)")
        print(f"Chunk Size: {arguments['chunk_size_tokens']} tokens")
        print(f"Vector DB: {arguments['vector_provider']}")
        print(f"Vector DB ID: {arguments['vector_db_id']}")
        print(f"Docling Service: Active")
        print("=" * 60)
        print("Pipeline Stages:")
        print("  1. Setup (Test & Prod)")
        print("  2. Batch Download (ALL files from bucket)")
        print("  3. Batch Docling Processing (each file)")
        print("  4. Vector DB Creation (Test & Prod)")
        print("  5. Batch Ingestion (all documents)")
        print("=" * 60)
        print("Monitor progress in the Kubeflow UI")