import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from groq import Groq

# Load environment variables
load_dotenv()

class MultiDocRAGSystem:
    def __init__(self):
        print("🚀 Initializing Kaleem's Professional Assistant\n")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        
        # Document configurations
        self.doc_config = {
            'transcript': {
                'collection': 'transcript',
                'processor': self.process_pdf,
                'description': 'academic transcript'
            },
            'resume': {
                'collection': 'resume',
                'processor': self.process_text,
                'description': 'professional resume'
            }
        }

        # System prompt template
        self.system_prompt = """You are Kaleem's professional assistant. Answer questions about his professional 
        experience, education, certifications, and academic record using ONLY the provided context. 

        Respond in FIRST PERSON when appropriate. Be precise and professional. 

        If information isn't available in the context, respond:
        "I don't have that information available. Would you like to know about Kaleem's {missing_info}?"

        Context:
        {context}

        Question: {question}
        Answer in MARKDOWN format:"""

        self._check_qdrant_connection()

    def _check_qdrant_connection(self):
        """Verify Qdrant connection"""
        try:
            print("🔍 Checking Qdrant connection...")
            self.qdrant_client.get_collections()
            print("✅ Connected to Qdrant successfully\n")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            raise

    def process_pdf(self, file_path: str) -> List[str]:
        """Process PDF documents with OCR"""
        print(f"\n📄 Processing PDF: {os.path.basename(file_path)}")
        images = convert_from_path(file_path)
        text = "\n\n".join([
            pytesseract.image_to_string(image)
            for image in images
        ])
        return self._chunk_text(text, "transcript")

    def process_text(self, file_path: str) -> List[str]:
        """Process text documents"""
        print(f"\n📝 Processing text file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._chunk_text(text, "resume")

    def _chunk_text(self, text: str, doc_type: str) -> List[str]:
        """Split text into chunks with document-specific parameters"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800 if doc_type == 'transcript' else 600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", "]
        )
        chunks = splitter.split_text(text)
        print(f"✂️ Created {len(chunks)} chunks for {doc_type}")
        return self._store_chunks(chunks, doc_type)

    def _store_chunks(self, chunks: List[str], doc_type: str) -> List[str]:
        """Store chunks in vector database"""
        Qdrant.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=self.doc_config[doc_type]['collection'],
            force_recreate=True
        )
        print(f"💾 Stored {len(chunks)} {doc_type} chunks in Qdrant\n")
        return chunks

    def load_documents(self):
        """Load and process all documents"""
        print("📂 Loading documents...")
        files = {
            'transcript': "BS CE Transcript.pdf",
            'resume': "resume.txt"
        }
        
        for doc_type, filename in files.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"❌ Missing {doc_type} file: {filename}")
            
            self.doc_config[doc_type]['processor'](filename)

    def _query_collection(self, query: str, doc_type: str, k: int = 3) -> List[str]:
        """Search a specific collection"""
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.doc_config[doc_type]['collection'],
            embeddings=self.embeddings
        )
        
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=0.45
        )
        
        return [
            f"[From {self.doc_config[doc_type]['description']}]\n{doc.page_content}"
            for doc, score in results
            if score > 0.45
        ]

    def get_context(self, query: str) -> Dict[str, List[str]]:
        """Get context from all document collections"""
        print(f"\n🔎 Searching for: '{query}'")
        context = {
            'transcript': self._query_collection(query, 'transcript'),
            'resume': self._query_collection(query, 'resume')
        }
        
        print("📚 Search results:")
        for doc_type, results in context.items():
            print(f"  - {doc_type}: {len(results)} relevant chunks")
        
        return context

    def generate_response(self, query: str) -> str:
        """Generate final response using Groq"""
        context_data = self.get_context(query)
        all_context = []
        
        for doc_type in ['resume', 'transcript']:
            all_context.extend(context_data[doc_type])
        
        if not all_context:
            available_info = " or ".join([
                cfg['description'] 
                for cfg in self.doc_config.values()
            ])
            return f"I don't have information about that. Would you like to know about Kaleem's {available_info}?"
        
        prompt = self.system_prompt.format(
                    context="\n\n".join(all_context)[:3000],
                    question=query,
                    missing_info=self.doc_config.get(list(self.doc_config.keys())[0], {}).get('description', 'documents')
                )
        
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

def main():
    assistant = MultiDocRAGSystem()
    
    try:
        assistant.load_documents()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    print("\n" + "="*50)
    print("🤖 Kaleem's Professional Assistant Ready!")
    print("You can ask about:")
    print("- Education history and academic performance")
    print("- Work experience and projects")
    print("- Technical skills and certifications")
    print("- Any other professional information\n")
    print("Type 'exit' to end the session")
    print("="*50 + "\n")
    
    while True:
        try:
            query = input("🧑💻 You: ")
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
                
            print("\n" + "━"*30)
            response = assistant.generate_response(query)
            print(f"\n🤖 Assistant:\n{response}")
            print("\n" + "━"*30 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Session ended")
            break

if __name__ == "__main__":
    main()