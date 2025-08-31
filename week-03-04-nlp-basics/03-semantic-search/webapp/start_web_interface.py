#!/usr/bin/env python3
"""
Semantic Search Web Interface Startup Script
============================================

Easy startup script for the semantic search web interface.
Handles initialization, dependency checking, and service startup.

Usage:
    python start_web_interface.py [--host HOST] [--port PORT] [--dev]

Options:
    --host HOST     Host to bind to (default: 0.0.0.0)
    --port PORT     Port to bind to (default: 8000)
    --dev           Enable development mode with auto-reload
    --check-only    Only check dependencies and service health

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'torch',
        'transformers',
        'sentence_transformers',
        'faiss-cpu',
        'scikit-learn',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install requirements:")
        print("pip install -r requirements_web.txt")
        return False
    
    print("[OK] All dependencies available")
    return True

def check_data_availability():
    """Check if search data is available."""
    print("\nChecking search data...")
    
    base_path = Path(__file__).parent
    demo_storage = base_path / "demo_storage"
    
    if not demo_storage.exists():
        print("[ERROR] Demo storage directory not found")
        print("Please run the search integration demo first:")
        print("python search_integration_demo.py")
        return False
    
    # Check for essential files/directories
    document_store = demo_storage / "document_store"
    embeddings_cache = demo_storage / "embeddings_cache.json"
    
    if not document_store.exists():
        print("[ERROR] Document store not found")
        return False
    
    if not embeddings_cache.exists():
        print("[ERROR] Embeddings cache not found")
        return False
    
    print("[OK] Document store available")
    print("[OK] Embeddings cache available")
    
    # Check for vector indices
    vector_indices = demo_storage / "vector_indices"
    if vector_indices.exists():
        print("[OK] Vector indices available")
    else:
        print("[WARN] Vector indices not found - will be built on startup")
    
    return True

def start_server(host="0.0.0.0", port=8000, dev_mode=False):
    """Start the FastAPI server."""
    print(f"\nStarting Semantic Search Web Interface...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Development mode: {dev_mode}")
    print("=" * 50)
    
    # Import and run the app
    try:
        import uvicorn
        
        # Run the server
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=dev_mode,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

def print_startup_info():
    """Print helpful startup information."""
    print("Semantic Search Engine Web Interface")
    print("=" * 50)
    print("Educational interface showcasing:")
    print("- Semantic search with vector similarity")
    print("- Classical TF-IDF search")
    print("- Hybrid ranking combining both approaches")
    print("- Performance comparison and analysis")
    print("- Real-time search statistics")
    print("=" * 50)

def print_access_info(host, port):
    """Print access information after successful startup."""
    print("\n" + "=" * 50)
    print("Server started successfully!")
    print("=" * 50)
    
    if host == "0.0.0.0":
        print(f"Local access: http://localhost:{port}")
        print(f"Network access: http://<your-ip>:{port}")
    else:
        print(f"Access URL: http://{host}:{port}")
    
    print(f"API Documentation: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/docs")
    print(f"Alternative docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Start the Semantic Search Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Enable development mode with auto-reload'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check dependencies and service health'
    )
    
    args = parser.parse_args()
    
    # Print startup information
    print_startup_info()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data availability
    if not check_data_availability():
        print("\n" + "=" * 50)
        print("[SETUP REQUIRED]")
        print("=" * 50)
        print("The semantic search system needs to be initialized first.")
        print("Please run the following command to set up the search indices:")
        print()
        print("    python search_integration_demo.py")
        print()
        print("This will:")
        print("- Process sample documents")
        print("- Build vector embeddings")
        print("- Create FAISS indices")
        print("- Build TF-IDF index")
        print("- Verify all components")
        print()
        
        if not args.check_only:
            response = input("Would you like to run the setup now? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("\nRunning search integration demo...")
                try:
                    subprocess.run([sys.executable, 'search_integration_demo.py'], check=True)
                    print("\n[OK] Setup completed successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Setup failed: {e}")
                    sys.exit(1)
                except FileNotFoundError:
                    print("[ERROR] search_integration_demo.py not found")
                    sys.exit(1)
            else:
                print("Setup skipped. Please run the demo script manually.")
                sys.exit(1)
        else:
            sys.exit(1)
    
    if args.check_only:
        print("\n[OK] All checks passed! Ready to start the web interface.")
        print("Run without --check-only to start the server.")
        return
    
    print("\n[OK] All checks passed!")
    print_access_info(args.host, args.port)
    
    # Start the server
    try:
        success = start_server(args.host, args.port, args.dev)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()