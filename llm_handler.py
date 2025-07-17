from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")


class LLMHandler:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize LLM handler.

        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        try:
            # For text generation, use a model that's better suited for QA
            if "DialoGPT" in model_name:
                # Use a better model for text generation
                self.model_name = "microsoft/DialoGPT-medium"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.generator = None
            else:
                # Use pipeline for other models
                self.generator = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.tokenizer = self.generator.tokenizer
                self.model = self.generator.model

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to distilgpt2...")
            self.model_name = "distilgpt2"
            self.generator = pipeline(
                "text-generation",
                model="distilgpt2",
                device=0 if self.device == "cuda" else -1
            )
            self.tokenizer = self.generator.tokenizer
            self.model = self.generator.model

    def generate_answer(self, query: str, context_chunks: List[Dict[str, str]],
                        max_length: int = 512) -> str:
        """
        Generate answer using retrieved context chunks.

        Args:
            query: User query
            context_chunks: List of relevant chunks
            max_length: Maximum length of generated response

        Returns:
            Generated answer
        """
        try:
            # Prepare context
            context = self._prepare_context(context_chunks)

            # Create prompt
            prompt = self._create_prompt(query, context)

            # Generate response
            if self.generator:
                response = self.generator(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                generated_text = response[0]['generated_text']
                # Extract only the new generated part
                answer = generated_text[len(prompt):].strip()

            else:
                # Use model directly for DialoGPT
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated_text[len(prompt):].strip()

            # Clean up the answer
            answer = self._clean_answer(answer)

            return answer if answer else "I couldn't generate a relevant answer based on the provided context."

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."

    def _prepare_context(self, context_chunks: List[Dict[str, str]],
                         max_context_length: int = 1500) -> str:
        """
        Prepare context from retrieved chunks.

        Args:
            context_chunks: List of chunk dictionaries
            max_context_length: Maximum characters for context

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for chunk in context_chunks:
            content = chunk['content']
            if current_length + len(content) <= max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # Add partial content if it fits
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful amount remains
                    context_parts.append(content[:remaining] + "...")
                break

        return " ".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following context, please answer the question.

Context: {context}

Question: {query}

Answer:"""

        return prompt

    def _clean_answer(self, answer: str) -> str:
        """
        Clean and format the generated answer.

        Args:
            answer: Raw generated answer

        Returns:
            Cleaned answer
        """
        # Remove extra whitespace
        answer = answer.strip()

        # Remove common artifacts
        answer = answer.replace("<|endoftext|>", "")
        answer = answer.replace("</s>", "")

        # Take only the first paragraph for cleaner output
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0]

        # Remove incomplete sentences at the end
        sentences = answer.split(".")
        if len(sentences) > 1 and sentences[-1].strip() and len(sentences[-1].strip()) < 10:
            answer = ".".join(sentences[:-1]) + "."

        return answer

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else "Unknown"
        }