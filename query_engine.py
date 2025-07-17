from typing import List, Dict, Tuple
import time
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_handler import LLMHandler


class QueryEngine:
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "microsoft/DialoGPT-medium",
                 chunk_size: int = 400,
                 overlap: int = 50):
        """
        Initialize the RAG query engine.

        Args:
            embedding_model: Name of the sentence transformer model
            llm_model: Name of the LLM model
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        print("Initializing RAG Query Engine...")

        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        self.vector_store = VectorStore(model_name=embedding_model)
        self.llm_handler = LLMHandler(model_name=llm_model)

        self.is_ready = False
        self.index_stats = {}

    def setup_from_documents(self, data_dir: str):
        """
        Setup the query engine by processing documents and building index.

        Args:
            data_dir: Directory containing documents
        """
        print(f"Setting up query engine from documents in: {data_dir}")

        # Load and process documents
        documents = self.doc_processor.load_documents(data_dir)
        if not documents:
            print("No documents loaded. Please check the data directory.")
            return False

        # Create chunks
        chunks = self.doc_processor.process_documents(documents)
        if not chunks:
            print("No chunks created from documents.")
            return False

        # Build vector index
        self.vector_store.build_index(chunks)

        # Update status
        self.is_ready = True
        self.index_stats = self.vector_store.get_stats()

        print("Query engine setup completed successfully!")
        return True

    def query(self, question: str, top_k: int = 5, verbose: bool = False) -> Dict:
        """
        Process a query and return answer with metadata.

        Args:
            question: User question
            top_k: Number of top chunks to retrieve
            verbose: Whether to print detailed information

        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_ready:
            return {
                "answer": "Query engine not ready. Please setup from documents first.",
                "error": "Not initialized"
            }

        start_time = time.time()

        try:
            # Step 1: Retrieve relevant chunks
            if verbose:
                print(f"Searching for relevant chunks for: '{question}'")

            search_results = self.vector_store.search(question, k=top_k)

            if not search_results:
                return {
                    "answer": "No relevant information found for your question.",
                    "retrieved_chunks": [],
                    "processing_time": time.time() - start_time
                }

            # Extract chunks and scores
            retrieved_chunks = [chunk for chunk, score in search_results]
            similarity_scores = [score for chunk, score in search_results]

            if verbose:
                print(f"Retrieved {len(retrieved_chunks)} chunks")
                for i, (chunk, score) in enumerate(search_results):
                    print(f"  Chunk {i + 1} (score: {score:.3f}): {chunk['content'][:100]}...")

            # Step 2: Generate answer using LLM
            if verbose:
                print("Generating answer using LLM...")

            answer = self.llm_handler.generate_answer(question, retrieved_chunks)

            processing_time = time.time() - start_time

            # Prepare response
            response = {
                "answer": answer,
                "question": question,
                "retrieved_chunks": [
                    {
                        "content": chunk['content'],
                        "filename": chunk.get('filename', 'unknown'),
                        "similarity_score": score
                    }
                    for chunk, score in search_results
                ],
                "processing_time": processing_time,
                "model_info": {
                    "embedding_model": self.vector_store.embedding_model,
                    "llm_model": self.llm_handler.model_name
                }
            }

            if verbose:
                print(f"Query processed in {processing_time:.2f} seconds")

            return response

        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def get_system_info(self) -> Dict:
        """
        Get information about the current system state.

        Returns:
            Dictionary with system information
        """
        return {
            "is_ready": self.is_ready,
            "index_stats": self.index_stats,
            "models": {
                "embedding": self.vector_store.embedding_model if hasattr(self.vector_store,
                                                                          'embedding_model') else "Not loaded",
                "llm": self.llm_handler.get_model_info()
            },
            "chunking_config": {
                "chunk_size": self.doc_processor.chunk_size,
                "overlap": self.doc_processor.overlap
            }
        }

    def save_index(self, filepath: str):
        """
        Save the vector index to disk.

        Args:
            filepath: Path to save the index
        """
        if self.is_ready:
            self.vector_store.save_index(filepath)
        else:
            print("Cannot save index - system not ready")

    def load_index(self, filepath: str):
        """
        Load a pre-built vector index from disk.

        Args:
            filepath: Path to load the index from
        """
        self.vector_store.load_index(filepath)
        if self.vector_store.index is not None:
            self.is_ready = True
            self.index_stats = self.vector_store.get_stats()
            print("Index loaded successfully")
        else:
            print("Failed to load index")

    def add_documents(self, data_dir: str):
        """
        Add more documents to existing index.

        Args:
            data_dir: Directory containing new documents
        """
        if not self.is_ready:
            print("System not ready. Use setup_from_documents first.")
            return

        print(f"Adding documents from: {data_dir}")

        # Load new documents
        new_documents = self.doc_processor.load_documents(data_dir)
        if not new_documents:
            print("No new documents found.")
            return

        # Create chunks from new documents
        new_chunks = self.doc_processor.process_documents(new_documents)
        if not new_chunks:
            print("No new chunks created.")
            return

        # Add to existing chunks and rebuild index
        all_chunks = self.vector_store.chunks + new_chunks
        self.vector_store.build_index(all_chunks)

        # Update stats
        self.index_stats = self.vector_store.get_stats()
        print(f"Added {len(new_chunks)} new chunks. Total: {len(all_chunks)}")