# File: rag_resume.py
import os
from dotenv import load_dotenv
from typing import List
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from groq import Groq

# Load environment variables
load_dotenv()

class ResumeRAGSystem:
    def __init__(self):
        print("Initializing RAG system...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
            collection_name="resume",
            embeddings=self.embeddings,
        )
        
        # System prompt template
        self.system_prompt = """You are a resume assistant that strictly answers questions based on the provided resume content. 
        If a question cannot be answered using the resume information, respond with:
        "I'm sorry, that information is not available in my resume. Would you like to know about something else?"
        
        Resume Content:
        {context}
        
        Question: {question}
        Answer:"""
        
        print("✓ RAG system initialized successfully\n")

    def process_resume(self, file_path: str) -> List[str]:
        """Process PDF resume with validation"""
        print(f"\nProcessing resume: {file_path}")
        
        # Validate PDF
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found at {file_path}")
        
        # Extract text
        print("Extracting text from PDF...")
        loader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in loader.pages])
        
        if not text.strip():
            raise ValueError("No text extracted from PDF! Check if PDF is text-based.")
            
        print(f"✓ Extracted {len(text)} characters")
        print("\nSample text (first 500 chars):")
        print(text[:500].replace("\n", " ") + "...\n")

        # Split text
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        print(f"✓ Created {len(chunks)} chunks")
        print("\nSample chunk (first 200 chars):")
        print(chunks[0][:200].replace("\n", " ") + "...\n")

        # Store vectors
        print("Storing vectors in Qdrant...")
        self.vector_store.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="resume",
            force_recreate=True,
        )
        
        # Verify storage
        collection_info = self.qdrant_client.get_collection("resume")
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
            score_threshold=0.4  # Temporarily lower for debugging
        )
        
        print(f"Found {len(results)} results:")
        for i, (doc, score) in enumerate(results):
            print(f"\nResult {i+1} (Score: {score:.2f}):")
            print(doc.page_content[:200].replace("\n", " ") + "...")
        
        return [doc.page_content for doc, score in results if score > 0.4]

    def generate_response(self, query: str) -> str:
        """Generate response with context validation"""
        context = self.get_context(query)
        
        if not context:
            print("\n⚠️ No relevant context found!")
            return "I'm sorry, that information is not available in my resume. Would you like to know about something else?"
        
        # Check if context actually contains answer
        print("\nContext being used:")
        for i, c in enumerate(context):
            print(f"{i+1}. {c[:200].replace('\n', ' ')}...")
        
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
    rag_system = ResumeRAGSystem()
    
    # Process resume (run only once)
    resume_path = "Resume___Engr__Muhammad_Kaleem_Ullah.pdf"
    if not os.path.exists(resume_path):
        print(f"❌ Resume file not found at {resume_path}")
        print("Please ensure the resume PDF is in the same directory as this script")
        return
    
    try:
        print("\nStarting resume processing...")
        chunks = rag_system.process_resume(resume_path)
        if not chunks:
            print("❌ No chunks created!")
            return
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        return
    
    # Interactive session
    print("\nResume Assistant: Ask me anything about my resume!")
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