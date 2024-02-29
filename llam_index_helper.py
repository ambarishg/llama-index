from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from config import *

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

client = QdrantClient(URL,port=6333)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

from llama_index.llms.azure_openai import AzureOpenAI
llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name=deployment_id_gpt4,
    api_key=key,
    azure_endpoint=endpoint,
    api_version=api_version,
)

from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding()


from llama_index.core import Settings
from llama_index.core  import VectorStoreIndex

Settings.llm = llm
Settings.embed_model = embed_model

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
Settings.llm = llm
Settings.embed_model = embed_model
# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    llm = llm,
)

# configure response synthesizer
# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)


def get_response(prompt):
    return query_engine.query(prompt)