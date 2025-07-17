# Mini LLM-Powered Question-Answering System (RAG)

A complete implementation of a Retrieval-Augmented Generation (RAG) system for document-based question answering.

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Add your documents**:
   - Place text documents (.txt files) in the `data/` directory

3. **Run the system**:
```bash
python main.py
```

4. **Optional: Launch web interface**:
```bash
python gradio_interface.py
```

## 📁 Project Structure

```
rag_qa_system/
├── src/
│   ├── document_processor.py    # Document loading and chunking
│   ├── vector_store.py         # Embedding generation and FAISS indexing
│   ├── llm_handler.py          # LLM integration for answer generation
│   └── query_engine.py         # Main RAG pipeline orchestration
├── data/                       # Directory for input documents
├── main.py                     # Command-line interface
├── test_queries.py            # Test suite for assignment queries
├── gradio_interface.py        # Web interface (bonus feature)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Implementation Details

### Core Components
- **Document Processing**: Sentence-based chunking with configurable overlap (400 tokens, 50 overlap)
- **Vector Store**: FAISS with `all-MiniLM-L6-v2` embeddings (384-dimensional)
- **LLM Handler**: DistilGPT2 for answer generation
- **Query Engine**: Complete RAG pipeline orchestration

### Dependencies
- `sentence-transformers==2.2.2` - Semantic embeddings
- `faiss-cpu==1.7.4` - Vector similarity search
- `transformers==4.35.0` - LLM integration
- `torch==2.1.0` - Neural network backend
- `gradio==3.50.0` - Web interface

## 🎯 Assignment Query Results

### Test Query 1
**Query**: "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission"
**System Response**: ✅ Successfully identifies F33.4 as the correct classification code

### Test Query 2
**Query**: "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?"
**System Response**: ✅ Provides comprehensive DSM-5 criteria with detailed explanations

## 🤖 AI Tool Usage

### Claude.ai Assistance
- **Primary Development Tool**: Used Claude.ai for most of the coding and debugging throughout the project
- **Code Structure**: Complete system architecture and module implementation
- **Documentation**: README generation and code documentation
- **Error Handling**: Debugging and exception handling implementation
- **Testing**: Test case design and validation logic
- **UI Development**: Gradio interface implementation and styling

### Development Challenges
- **Account Limitations**: Hit Claude.ai free subscription limits 4 times, requiring switching between different accounts
- **Code Continuity**: Had to restart conversations multiple times, making code more compact and streamlined
- **Time Management**: Lost significant time on UI development and account switching

## 📊 Performance Metrics

- **Setup Time**: ~30-60 seconds
- **Query Response**: 1-3 seconds average
- **Memory Usage**: ~500MB-1GB
- **Chunk Processing**: ~100-200 chunks/second

## ⚠️ Limitations and Project Constraints

### Time Constraints (4-hour limit)
- **Time Allocation**: 3 hours coding + 30 minutes documentation + 30 minutes video
- **UI Focus**: Spent excessive time on Gradio interface development
- **Account Issues**: Multiple Claude.ai account switches disrupted workflow
- **Completion Status**: Core functionality implemented but couldn't deliver 100% potential due to time constraints

### Technical Limitations
- **Document Formats**: Only supports plain text (.txt) files
- **Model Constraints**: Used smaller models for faster inference
- **Context Length**: Limited by LLM context window (~1000 tokens)

## 🚀 Bonus Features Implemented

- **Web Interface**: Interactive Gradio UI with real-time processing
- **Index Persistence**: Save/load pre-built indices
- **Comprehensive Testing**: Automated test suite with performance metrics
- **Source Visualization**: Shows retrieved chunks and similarity scores

## 📋 Running Tests

```bash
# Run assignment-specific tests
python test_queries.py

# Run interactive system
python main.py

# Launch web interface
python gradio_interface.py
```

## 📈 Project Summary

This RAG system successfully implements document-based question answering with semantic search and LLM integration. While the core functionality meets all assignment requirements, the development process was significantly impacted by:

- **Heavy reliance on Claude.ai** for coding and debugging
- **Account limitations** forcing multiple conversation restarts
- **Time mismanagement** focusing too much on UI development
- **Technical constraints** limiting the system's full potential

The final implementation demonstrates a working RAG pipeline but represents a compressed development cycle rather than optimal engineering practices.

## 📄 Assignment Details

**Completed By**: Meet P Barasara  
**Development Tool**: Primarily Claude.ai  
**Time Spent**: 4 hours total (3 hours coding + 1 hour documentation/video)  
**Status**: ✅ Core requirements met with bonus features, but could have achieved more with better time management