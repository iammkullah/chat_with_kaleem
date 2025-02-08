import os
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from groq import Groq
from transformers import ViTImageProcessor, ViTModel
import torch

# Load environment variables
load_dotenv()

class MultiModalRAGSystem:
    def __init__(self):
        print("🚀 Initializing Kaleem's Professional Assistant\n")
        
        # Initialize text embeddings
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vision model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(self.device)
        
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
                'description': 'academic transcript',
                'modality': 'text'
            },
            'resume': {
                'collection': 'resume',
                'processor': self.process_text,
                'description': 'professional resume',
                'modality': 'text'
            },
            'degree': {
                'collection': 'degree',
                'processor': self.process_image,
                'description': 'BS degree document',
                'modality': 'image'
            }
        }

        # System prompt template
        self.system_prompt = """You are Kaleem's professional assistant. Analyze documents and images to answer 
        questions about his professional credentials. When asked about visual elements in documents:
        1. Count visible elements like logos/signatures
        2. Describe spatial relationships
        3. Identify document security features
        4. Mention any special formatting

        Respond in FIRST PERSON when appropriate. Be precise and professional.

        Available context:
        {context}

        Question: {question}
        Answer in MARKDOWN format:"""

        self._check_qdrant_connection()
        self._init_collections()

    def _check_qdrant_connection(self):
        """Verify Qdrant connection"""
        try:
            print("🔍 Checking Qdrant connection...")
            self.qdrant_client.get_collections()
            print("✅ Connected to Qdrant successfully\n")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            raise

    def _init_collections(self):
        """Ensure collections exist with proper configuration"""
        for doc_type, config in self.doc_config.items():
            try:
                self.qdrant_client.get_collection(config['collection'])
            except:
                print(f"⚠️ Creating new collection: {config['collection']}")
                vector_size = 768 if config['modality'] == 'image' else 1024
                self.qdrant_client.create_collection(
                    collection_name=config['collection'],
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )

    def process_pdf(self, file_path: str) -> List[str]:
        """Process PDF documents with OCR"""
        print(f"\n📄 Processing PDF: {os.path.basename(file_path)}")
        images = convert_from_path(file_path)
        text = "\n\n".join([
            pytesseract.image_to_string(image)
            for image in images
        ])
        return self._process_text_content(text, "transcript")

    def process_text(self, file_path: str) -> List[str]:
        """Process text documents"""
        print(f"\n📝 Processing text file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._process_text_content(text, "resume")

    def process_image(self, file_path: str) -> List[np.ndarray]:
        """Process image documents directly to embeddings"""
        print(f"\n🖼️ Processing image: {os.path.basename(file_path)}")
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            self._store_image_embeddings(embeddings, "degree", file_path)
            return embeddings
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return []

    def _process_text_content(self, text: str, doc_type: str) -> List[str]:
        """Handle text processing pipeline"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800 if doc_type == 'transcript' else 600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", "]
        )
        chunks = splitter.split_text(text)
        print(f"✂️ Created {len(chunks)} chunks for {doc_type}")
        self._store_text_chunks(chunks, doc_type)
        return chunks

    def _store_text_chunks(self, chunks: List[str], doc_type: str) -> List[str]:
        """Store chunks in vector database"""
        Qdrant.from_texts(
            texts=chunks,
            embedding=self.text_embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=self.doc_config[doc_type]['collection'],
            force_recreate=True
        )
        print(f"💾 Stored {len(chunks)} {doc_type} chunks in Qdrant\n")
        return chunks

    def _store_image_embeddings(self, embeddings: np.ndarray, doc_type: str, source: str):
        """Store image embeddings directly in Qdrant"""
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "source": source,
                    "doc_type": doc_type,
                    "content_type": "image"
                }
            )
            for idx, embedding in enumerate(embeddings)
        ]
        self.qdrant_client.upsert(
            collection_name=self.doc_config[doc_type]['collection'],
            points=points
        )
        print(f"💾 Stored {len(embeddings)} image embeddings for {doc_type}\n")

    def load_documents(self):
        """Load and process all documents"""
        print("📂 Loading documents...")
        files = {
            'transcript': "BS CE Transcript.pdf",
            'resume': "resume.txt",
            'degree': "BS Degree.jpg"
        }
        
        for doc_type, filename in files.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"❌ Missing {doc_type} file: {filename}")
            
            print(f"\n🔨 Processing {doc_type.upper()} document")
            self.doc_config[doc_type]['processor'](filename)
            print(f"✅ Completed {doc_type.upper()} processing")

    def _query_text(self, query: str, doc_type: str, k: int = 3) -> List[str]:
        """Search text collections"""
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.doc_config[doc_type]['collection'],
            embeddings=self.text_embeddings
        )
        results = vector_store.similarity_search_with_score(query, k=k)
        return [
            f"[From {self.doc_config[doc_type]['description']}]\n{doc.page_content}"
            for doc, score in results
            if score > 0.45
        ]

    def _query_image(self, query: str, doc_type: str, k: int = 2) -> List[str]:
        """Search image collections using text-to-image cross-modal search"""
        try:
            # Convert text query to embedding using text model
            text_embedding = self.text_embeddings.embed_query(query)
            
            # Search image collection
            results = self.qdrant_client.search(
                collection_name=self.doc_config[doc_type]['collection'],
                query_vector=text_embedding,
                limit=k
            )
            
            return [
                f"[Visual match from {self.doc_config[doc_type]['description']}]\n" 
                f"Relevance score: {result.score:.2f}"
                for result in results
            ]
        except Exception as e:
            print(f"⚠️ Image search error: {e}")
            return []

    def get_context(self, query: str) -> Dict[str, List[str]]:
        """Get multimodal context for query"""
        print(f"\n🔎 Multimodal search for: '{query}'")
        context = {
            'text': [],
            'images': []
        }
        
        # Search text documents
        for doc_type in ['transcript', 'resume']:
            context['text'].extend(self._query_text(query, doc_type))
        
        # Search image documents
        context['images'].extend(self._query_image(query, 'degree'))
        
        print("📚 Search results:")
        print(f"  - Text matches: {len(context['text'])}")
        print(f"  - Visual matches: {len(context['images'])}")
        
        return context

    def generate_response(self, query: str) -> str:
        """Generate final response using Groq"""
        context = self.get_context(query)
        all_context = context['text'] + context['images']
        
        if not all_context:
            available_info = " or ".join([cfg['description'] for cfg in self.doc_config.values()])
            return f"I don't have relevant information. Would you like to know about Kaleem's {available_info}?"
        
        prompt = self.system_prompt.format(
            context="\n\n".join(all_context)[:3500],
            question=query
        )
        
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=1024,
                stop=["\n\n"]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

def main():
    assistant = MultiModalRAGSystem()
    
    try:
        assistant.load_documents()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    print("\n" + "="*60)
    print("🤖 Kaleem's Multimodal Professional Assistant Ready!")
    print("You can ask about:")
    print("- Educational qualifications (including document visual analysis)")
    print("- Professional experience and technical skills")
    print("- Visual elements in documents (logos, signatures, formatting)")
    print("- Academic performance and certifications")
    print("\nType 'exit' to end the session")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("🧑💻 You: ")
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
                
            print("\n" + "━"*40)
            response = assistant.generate_response(query)
            print(f"\n🤖 Assistant:\n{response}")
            print("\n" + "━"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Session ended")
            break

if __name__ == "__main__":
    main()