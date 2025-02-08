import os
import time
import numpy as np
from dotenv import load_dotenv
from typing import List, Union, Dict
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from groq import Groq
import torch
from transformers import CLIPProcessor, CLIPModel

# Import the LangChain Embeddings base class
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()


#########################################
#       CLIP EMBEDDINGS CLASS           #
#########################################
class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_query(self, text: str) -> List[float]:
        # Use truncation and max_length to avoid token length issues.
        inputs = self.processor(
            text=[text],
            truncation=True,
            max_length=77,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features[0].cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(
            text=texts,
            truncation=True,
            max_length=77,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy().tolist()

    def embed_image(self, image: Image.Image) -> List[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features[0].cpu().numpy().tolist()

    def __call__(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Allows the embedding object to be called. If a single string is provided,
        returns the query embedding; if a list of strings is provided, returns the document embeddings.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)


#########################################
#  MULTIMODAL RAG SYSTEM USING CLIP     #
#########################################
class MultiModalRAGSystem:
    def __init__(self):
        print("🚀 Initializing Kaleem's Professional Assistant\n")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP embeddings (for both text and image)
        self.clip_embeddings = CLIPEmbeddings(device=self.device)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        
        # Document configurations (all collections will use 512-d embeddings with CLIP)
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
                'description': 'degree certificate',
                'modality': 'image'
            }
        }

        # Updated system prompt with clear instructions
        self.system_prompt = (
            "You are Kaleem's professional assistant. You have access to context extracted from three documents:\n"
            "1. An academic transcript\n"
            "2. A professional resume\n"
            "3. A degree certificate (image)\n\n"
            "When a question is asked, analyze the provided context and answer the question as clearly and precisely as possible. "
            "Reference specific parts of the context if relevant. If the context does not provide a clear answer, state that the "
            "information is not available.\n\n"
            "Available context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer in MARKDOWN format:"
        )

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
        """Ensure collections exist with proper configuration.
        If a collection exists but its vector dimension does not match 512,
        delete and recreate it. Also, if create_collection raises a conflict,
        we catch it and continue.
        """
        expected_dim = 512
        for doc_type, config in self.doc_config.items():
            collection_name = config['collection']
            recreate = False
            try:
                collection_info = self.qdrant_client.get_collection(collection_name)
                existing_dim = collection_info.result.config.vectors.size
                if existing_dim != expected_dim:
                    print(f"⚠️ Collection '{collection_name}' has dimension {existing_dim} but expected {expected_dim}. Deleting and recreating...")
                    self.qdrant_client.delete_collection(collection_name)
                    time.sleep(1)
                    recreate = True
            except Exception as e:
                # If get_collection fails, assume the collection doesn't exist.
                recreate = True

            if recreate:
                try:
                    print(f"⚠️ Creating new collection: {collection_name}")
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=expected_dim,
                            distance=models.Distance.COSINE
                        )
                    )
                except Exception as e:
                    # If the error is due to a conflict, ignore it.
                    if "already exists" in str(e):
                        print(f"Collection '{collection_name}' already exists. Continuing...")
                    else:
                        raise e

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
        """Process image documents directly to embeddings using CLIP"""
        print(f"\n🖼️ Processing image: {os.path.basename(file_path)}")
        try:
            image = Image.open(file_path).convert("RGB")
            embedding = self.clip_embeddings.embed_image(image)
            embeddings = np.array([embedding])  # Wrap in an array for consistency
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
        """Store chunks in vector database using CLIP embeddings"""
        Qdrant.from_texts(
            texts=chunks,
            embedding=self.clip_embeddings,
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
        """Search text collections using CLIP embeddings"""
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.doc_config[doc_type]['collection'],
            embeddings=self.clip_embeddings
        )
        results = vector_store.similarity_search_with_score(query, k=k)
        return [
            f"[From {self.doc_config[doc_type]['description']}]\n{doc.page_content}"
            for doc, score in results
            if score > 0.45
        ]

    def _query_image(self, query: str, doc_type: str, k: int = 2) -> List[str]:
        """Search image collections using text-to-image cross-modal search with CLIP"""
        try:
            text_embedding = self.clip_embeddings.embed_query(query)
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
        for doc_type in ['transcript', 'resume']:
            context['text'].extend(self._query_text(query, doc_type))
        context['images'].extend(self._query_image(query, 'degree'))
        print("📚 Search results:")
        print(f"  - Text matches: {len(context['text'])}")
        print(f"  - Visual matches: {len(context['images'])}")
        return context

    def generate_response(self, query: str) -> str:
        """Generate final response using Groq"""
        context = self.get_context(query)
        # Combine context (text first, then images) and truncate if needed.
        all_context = "\n\n".join(context['text'] + context['images'])[:3500]
        prompt = self.system_prompt.format(
            context=all_context,
            question=query
        )
        print("\n=== Generated Prompt ===")
        print(prompt)
        print("========================\n")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=1024
                # Removed explicit stop tokens to allow full responses.
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
