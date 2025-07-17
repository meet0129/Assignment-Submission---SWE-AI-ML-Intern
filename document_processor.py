import os
from typing import List, Dict
import re


class DocumentProcessor:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        """
        Initialize document processor with chunking parameters.

        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_documents(self, data_dir: str) -> List[Dict[str, str]]:
        """
        Load all text documents from the data directory.

        Args:
            data_dir: Path to directory containing documents

        Returns:
            List of document dictionaries with 'content' and 'filename' keys
        """
        documents = []

        if not os.path.exists(data_dir):
            print(f"Warning: Data directory {data_dir} does not exist")
            return documents

        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'content': content,
                            'filename': filename
                        })
                        print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        return documents

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters for English).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.

        Strategy:
        1. Split by sentences first to maintain semantic coherence
        2. Group sentences into chunks of approximately chunk_size tokens
        3. Add overlap between consecutive chunks

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with each chunk

        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []

        # Split into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_dict = {
                    'content': chunk_text,
                    'token_count': current_tokens
                }
                if metadata:
                    chunk_dict.update(metadata)
                chunks.append(chunk_dict)

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.estimate_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_dict = {
                'content': chunk_text,
                'token_count': current_tokens
            }
            if metadata:
                chunk_dict.update(metadata)
            chunks.append(chunk_dict)

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get sentences for overlap based on token count.

        Args:
            sentences: List of sentences from previous chunk

        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        overlap_tokens = 0

        # Take sentences from the end until we reach overlap limit
        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def process_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process all documents into chunks.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunk dictionaries
        """
        all_chunks = []

        for doc in documents:
            metadata = {'filename': doc['filename']}
            chunks = self.chunk_text(doc['content'], metadata)
            all_chunks.extend(chunks)
            print(f"Created {len(chunks)} chunks from {doc['filename']}")

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks