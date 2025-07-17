#!/usr/bin/env python3

import gradio as gr
import os
import json
from src.query_engine import QueryEngine


class GradioInterface:
    def __init__(self):
        """Initialize the Gradio interface."""
        self.query_engine = None
        self.setup_status = "Not initialized"

    def setup_system(self, progress=gr.Progress()):
        """Setup the RAG system."""
        try:
            progress(0.1, desc="Initializing query engine...")

            self.query_engine = QueryEngine(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                chunk_size=400,
                overlap=50
            )

            progress(0.3, desc="Loading documents...")

            # Check if data directory exists
            if not os.path.exists("data") or not os.listdir("data"):
                self.setup_status = "Error: No documents found in data/ directory"
                return self.setup_status

            progress(0.5, desc="Processing documents and building index...")

            success = self.query_engine.setup_from_documents("data")

            if success:
                progress(1.0, desc="Setup completed!")
                self.setup_status = "✅ System ready for queries"
                return self.setup_status
            else:
                self.setup_status = "❌ Failed to setup system"
                return self.setup_status

        except Exception as e:
            self.setup_status = f"❌ Error: {str(e)}"
            return self.setup_status

    def query_system(self, question, top_k, show_details):
        """Process a user query."""
        if self.query_engine is None or not self.query_engine.is_ready:
            return "❌ System not ready. Please setup the system first.", "", ""

        if not question.strip():
            return "Please enter a question.", "", ""

        try:
            # Process query
            result = self.query_engine.query(question, top_k=top_k, verbose=False)

            # Format main answer
            answer = result['answer']

            # Format metadata
            metadata = f"⏱️ Processing time: {result['processing_time']:.2f} seconds\n"
            metadata += f"📊 Retrieved {len(result.get('retrieved_chunks', []))} chunks\n"
            metadata += f"🤖 Model: {result.get('model_info', {}).get('llm_model', 'Unknown')}"

            # Format retrieved chunks if requested
            chunks_info = ""
            if show_details and 'retrieved_chunks' in result:
                chunks_info = "📋 **Retrieved Chunks:**\n\n"
                for i, chunk in enumerate(result['retrieved_chunks'], 1):
                    chunks_info += f"**Chunk {i}** (Score: {chunk['similarity_score']:.3f})\n"
                    chunks_info += f"Source: {chunk['filename']}\n"
                    chunks_info += f"Content: {chunk['content'][:200]}...\n\n"

            return answer, metadata, chunks_info

        except Exception as e:
            return f"❌ Error processing query: {str(e)}", "", ""

    def get_system_info(self):
        """Get system information."""
        if self.query_engine is None:
            return "System not initialized"

        try:
            info = self.query_engine.get_system_info()
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error getting system info: {str(e)}"

    def create_interface(self):
        """Create the Gradio interface."""

        with gr.Blocks(title="RAG QA System", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Mini LLM-Powered Question-Answering System (RAG)")
            gr.Markdown("Upload documents to `data/` directory and setup the system to start asking questions.")

            with gr.Tab("Query System"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Enter your question",
                            placeholder="e.g., What are the diagnostic criteria for OCD?",
                            lines=2
                        )

                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Number of chunks to retrieve"
                            )

                            show_details_checkbox = gr.Checkbox(
                                label="Show retrieved chunks",
                                value=False
                            )

                        query_btn = gr.Button("Ask Question", variant="primary")

                    with gr.Column(scale=1):
                        setup_btn = gr.Button("Setup System", variant="secondary")
                        setup_status = gr.Textbox(
                            label="Setup Status",
                            value="Not initialized",
                            interactive=False
                        )

                # Output sections
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False
                )

                metadata_output = gr.Textbox(
                    label="Query Metadata",
                    lines=3,
                    interactive=False
                )

                chunks_output = gr.Markdown(
                    label="Retrieved Chunks",
                    visible=False
                )

                # Event handlers
                setup_btn.click(
                    fn=self.setup_system,
                    outputs=[setup_status]
                )

                query_btn.click(
                    fn=self.query_system,
                    inputs=[question_input, top_k_slider, show_details_checkbox],
                    outputs=[answer_output, metadata_output, chunks_output]
                )

                # Show/hide chunks based on checkbox
                show_details_checkbox.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[show_details_checkbox],
                    outputs=[chunks_output]
                )

            with gr.Tab("System Info"):
                system_info_btn = gr.Button("Get System Info")
                system_info_output = gr.Code(
                    label="System Information",
                    language="json",
                    lines=20
                )

                system_info_btn.click(
                    fn=self.get_system_info,
                    outputs=[system_info_output]
                )

            with gr.Tab("Sample Queries"):
                gr.Markdown("## Assignment Test Queries")

                sample_queries = [
                    "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission",
                    "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?",
                    "What is the difference between obsessions and compulsions?",
                    "How is remission defined in depression?",
                    "What are the treatment options for recurrent depression?"
                ]

                for i, query in enumerate(sample_queries, 1):
                    with gr.Row():
                        gr.Textbox(
                            value=query,
                            label=f"Sample Query {i}",
                            interactive=False,
                            scale=4
                        )
                        copy_btn = gr.Button("Copy", scale=1)

                        # Note: In a real implementation, you'd need JavaScript to copy to clipboard
                        # For now, this is just a placeholder

            with gr.Tab("Instructions"):
                gr.Markdown("""
                ## How to Use This System

                1. **Setup Documents**: Place your text documents (.txt files) in the `data/` directory

                2. **Initialize System**: Click "Setup System" to process documents and build the search index

                3. **Ask Questions**: Enter your question and click "Ask Question"

                4. **Adjust Parameters**: 
                   - Use the slider to control how many document chunks to retrieve
                   - Check "Show retrieved chunks" to see the source material used for answering

                ## Features

                - **Document Processing**: Automatically chunks documents for optimal retrieval
                - **Semantic Search**: Uses sentence transformers for finding relevant content
                - **LLM Generation**: Generates answers using a local language model
                - **Source Tracking**: Shows which documents were used to generate answers

                ## Supported Formats

                - Plain text files (.txt)
                - Documents should be placed in the `data/` directory

                ## Performance Notes

                - Initial setup may take a few minutes depending on document size
                - Query processing typically takes 1-3 seconds
                - System works best with well-structured documents
                """)

        return interface


def main():
    """Main function to launch the Gradio interface."""

    # Create interface instance
    interface_handler = GradioInterface()

    # Create and launch interface
    interface = interface_handler.create_interface()

    interface.queue()
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Set to True for public sharing
        debug=False,
        show_error=True
    )


if __name__ == "__main__":
    main()