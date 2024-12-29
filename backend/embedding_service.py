import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
import csv
from dotenv import load_dotenv, find_dotenv
from typing import List
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
class FAQEmbeddingService:
    """A service for embedding FAQ data using OpenAI and Chroma vector store.

    This service handles:
    - Reading Q&A pairs from CSV files
    - Embedding the text using OpenAI
    - Storing and retrieving embeddings using Chroma
    """
    
    # Class constants for configuration
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_COLLECTION_NAME = "product_faq_collection"


    
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):

        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )
        self.llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model_name="gpt-3.5-turbo"
        )

    def ingest_faq_csv(
        self,
        csv_file_path: str,
        question_column: str = "Question",
        answer_column: str = "Answer",
        delimiter: str = ","
    ) -> None:
        """Ingest FAQ data from a CSV file into the vector store."""
        docs = []
        with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                question = row["Question"]
                answer = row["Answer"]
                combined_text = f"Q: {question}\nA: {answer}"
                docs.append(Document(page_content=combined_text))

        # Optional chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splitted_docs = []
        for doc in docs:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                splitted_docs.append(Document(page_content=chunk))

        # Add to Chroma
        self.vectorstore.add_documents(splitted_docs)
        self.vectorstore.persist()

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[Document]:
        """Get relevant chunks for a given query."""
        logger = logging.getLogger(__name__)
        logger.info(f"Searching for query: {query}")
        
        try:
            # Use similarity_search_with_score instead of similarity_search
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results_with_scores)} results with scores")
            
            # Convert results to Documents with scores in metadata
            documents = []
            for doc, score in results_with_scores:
                # Convert score to similarity score (assuming cosine distance)
                similarity_score = 1 - score  # Convert distance to similarity
                doc.metadata['score'] = similarity_score
                documents.append(doc)
                logger.info(f"Document score: {similarity_score}")
                
            return documents
            
        except Exception as e:
            logger.error(f"Error in get_relevant_chunks: {str(e)}")
            raise

    def get_qa_chain(self) -> RetrievalQA:
        """Get the QA chain for answering questions."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create the chain with specific configuration
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,  # Don't return source documents
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know.

                    Context: {context}

                    Question: {question}
                    
                    Answer: """,
                    input_variables=["context", "question"]
                )
            }
        )
        
        return chain
