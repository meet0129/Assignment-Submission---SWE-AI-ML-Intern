import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import os


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with embedding model.

        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.embeddings = None

    def embed_chunks(self, chunks: List[Dict[str, str]]) -> np.ndarray:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def build_index(self, chunks: List[Dict[str, str]]):
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dictionaries
        """
        if not chunks:
            print("No chunks provided for indexing")
            return

        self.chunks = chunks
        self.embeddings = self.embed_chunks(chunks)

        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Add embeddings to index
        self.index.add(self.embeddings)

        print(f"Index built successfully with {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for similar chunks to the query.

        Args:
            query: Search query
            k: Number of top results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index is None:
            print("Index not built yet. Call build_index() first.")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search in index
        similarities, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.chunks):  # Ensure valid index
                chunk = self.chunks[idx]
                results.append((chunk, float(similarity)))

        return results

    def save_index(self, filepath: str):
        """
        Save the vector store to disk.

        Args:
            filepath: Path to save the index
        """
        if self.index is None:
            print("No index to save")
            return

        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")

        # Save chunks and embeddings
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'dimension': self.dimension
            }, f)

        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """
        Load the vector store from disk.

        Args:
            filepath: Path to load the index from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")

            # Load chunks and embeddings
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
                self.dimension = data['dimension']

            print(f"Index loaded from {filepath}")
            print(f"Loaded {len(self.chunks)} chunks")

        except FileNotFoundError:
            print(f"Index files not found at {filepath}")
        except Exception as e:
            print(f"Error loading index: {e}")

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        if self.index is None:
            return {"status": "Index not built"}

        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.dimension,
            "index_size": self.index.ntotal,
            "model_name": self.embedding_model.get_sentence_embedding_dimension()
        }