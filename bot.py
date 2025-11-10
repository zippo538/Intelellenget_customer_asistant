from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from typing import List
import os
import getpass
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


class Rag:
    def __init__(self,file_path :str, collection_name :str) :
        self.model_llm : str = "meta-llama/llama-4-maverick-17b-128e-instruct"
        self.url_qdrant : str = "http://localhost:6333"
        self.collection_name : str = collection_name
        self.file_path : str = file_path
        self.model_embeddding : str = "all-MiniLM-L6-v2"
        
         
         
    def _load_llm(self,temperature):
        llm = ChatGroq(
            model=self.model_llm,
            temperature=temperature,
            api_key=api_key,
            max_tokens=1024,
            timeout=None,
            max_retries=3
        )
        return llm

    # document laoder
    @staticmethod
    def extract_text_from_pdf(self) -> List[Document]:
        try : 
            reader= PyMuPDFLoader(self.file_path)
            docs = reader.load()
            print(f"✅ Berhasil memproses Dokumen")
            return docs
        except Exception as e:
            print(f"Error saat memproses PDF : {str(e)}")
            return []
        
    #chunking text
    @staticmethod
    def chunk_text(documents : List[Document],chunk_size=1000,overlap=200) -> List[Document]:
        print("Memulai Chunking")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", '\n●','\n1','\n2','\n3','\n4','\n5','\n6','\n7','\n8','\n9','\n10'," "]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Chunking selesai. Total halaman asli : {len(documents)}")
        return chunks

    #index to qdrant
    @staticmethod
    def index_to_qdrant(self, chunks : List[Document])-> None:
        if not chunks:
            print("❌ Tidak ada chunks yang valid untuk di-index.")
            return

        try:
            # Inisialisasi model embedding yang akan digunakan
            model = SentenceTransformerEmbeddings(model_name=self.model_embeddding)

            print(f"⏳ Memulai indexing ke Qdrant Collection: {self.collection_name}...")
            
            # 2. Indexing Otomatis
            # Qdrant.from_documents secara otomatis menangani:
            # a) Mengubah chunks menjadi embeddings.
            # b) Menyimpan embeddings, teks, dan metadata ke Qdrant.
            Qdrant.from_documents(
                documents=chunks,
                embedding=model,
                url = self.url_qdrant,
                collection_name=self.collection_name,
            )
            
            print("✅ Indexing ke Qdrant Selesai!")

        except Exception as e:
            print(f"❌ Gagal Indexing ke Qdrant. Cek koneksi atau API Key Anda. Error: {e}")
    
    
    def run_all(self) -> None :
        try :
            text_book = self.extract_text_from_pdf()
            chunk = self.chunk_text(text_book)
            
            self.index_to_qdrant(chunk)
            print("Berhasil memasukkan dalam index Qdrant")
            
        except Exception as e :
            print(f"Error running RAG : {str(e)}")
        

rag = Rag('Handbook_BNI_Mbank.pdf',"user_manual_BNI_Mbank")
rag.run_all()


