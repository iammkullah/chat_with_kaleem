import os
from dotenv import load_dotenv
from typing import List
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from groq import Groq

# Load environment variables
load_dotenv()

class TranscriptRAGSystem:
    def __init__(self):
        print("Initializing Transcript RAG system...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embeddings model loaded")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        
        # Check Qdrant connection
        try:
            print("Checking Qdrant connection...")
            self.qdrant_client.get_collections()
            print("✓ Connected to Qdrant successfully")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise

        # Initialize vector store
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name="transcript",
            embeddings=self.embeddings,
        )
        
        # System prompt template
        self.system_prompt = """You are a transcript assistant that strictly answers questions based on the provided transcript content. 
        If a question cannot be answered using the transcript information, respond with:
        "I'm sorry, that information is not available in the transcript. Would you like to know about something else?"
        
        Transcript Content:
        {context}
        
        Question: {question}
        Answer:"""
        
        print("✓ RAG system initialized successfully\n")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR"""
        print(f"\nProcessing PDF file: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path)
        print(f"✓ Converted {len(images)} pages")
        
        # Extract text from each image using OCR
        print("Performing OCR on images...")
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            extracted_text += f"\n\nPage {i+1}:\n{text}"
            print(f"✓ Processed page {i+1}")
        
        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the PDF!")
            
        print(f"\n✓ Extracted {len(extracted_text)} characters")
        print("\nSample text (first 500 chars):")
        print(extracted_text[:500].replace("\n", " ") + "...\n")
        
        return extracted_text

    def process_transcript(self, pdf_path: str) -> List[str]:
        """Process transcript PDF with OCR and chunking"""
        # Extract text using OCR
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split text into chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", "]
        )
        chunks = text_splitter.split_text(text)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Store vectors
        print("Storing vectors in Qdrant...")
        self.vector_store.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="transcript",
            force_recreate=True,
        )
        
        # Verify storage
        collection_info = self.qdrant_client.get_collection("transcript")
        print("\n✓ Vectors stored successfully")
        print(f"Collection info: {collection_info}")
        print(f"Total vectors stored: {collection_info.points_count}")
        
        return chunks

    def get_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve context with debugging"""
        print(f"\nSearching for: '{query}'")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=0.5
        )
        
        print(f"Found {len(results)} results:")
        for i, (doc, score) in enumerate(results):
            print(f"\nResult {i+1} (Score: {score:.2f}):")
            print(doc.page_content[:200].replace("\n", " ") + "...")
        
        return [doc.page_content for doc, score in results if score > 0.5]

    def generate_response(self, query: str) -> str:
        """Generate response with context validation"""
        context = self.get_context(query)
        
        if not context:
            print("\n⚠️ No relevant context found!")
            return "I'm sorry, that information is not available in the transcript. Would you like to know about something else?"
        
        # Prepare prompt
        prompt = self.system_prompt.format(
            context="\n\n".join(context),
            question=query
        )
        
        # Generate response
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=1024
        )
        
        return response.choices[0].message.content

def main():
    # Initialize RAG system
    rag_system = TranscriptRAGSystem()
    
    # Process transcript PDF file (run only once)
    transcript_path = "BS CE Transcript.pdf"  # Change to your PDF file name
    if not os.path.exists(transcript_path):
        print(f"❌ Transcript PDF file not found at {transcript_path}")
        print("Please add your transcript PDF in the same directory")
        return
    
    try:
        print("\nStarting transcript processing...")
        chunks = rag_system.process_transcript(transcript_path)
        if not chunks:
            print("❌ No chunks created!")
            return
    except Exception as e:
        print(f"❌ Error processing transcript: {e}")
        return
    
    # Interactive session
    print("\nTranscript Assistant: Ask me anything about the transcript!")
    print("Type 'exit' or 'quit' to end the session\n")
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                break
            response = rag_system.generate_response(query)
            print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()