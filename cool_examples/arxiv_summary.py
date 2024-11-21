# %% [markdown]
# # arXiv Research Assistant with Semantic Kernel
# 
# This notebook demonstrates how to:
# 1. Fetch latest papers from arXiv
# 2. Store abstracts in a semantic memory
# 3. Generate a research summary
# 
# First, let's import our dependencies:

# %%
import asyncio
import arxiv
from typing import List

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.functions import KernelFunctionFromPrompt

# %% [markdown]
# ## Setup Kernel and Memory
# 
# Initialize our semantic kernel with OpenAI services and setup memory:

# %%
async def setup_kernel_and_memory():
    kernel = Kernel()
    
    # Add chat completion service
    service_id = "chat-gpt"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-3.5-turbo"
        )
    )

    # Add embedding service
    embedding_gen = OpenAITextEmbedding(
        service_id="ada",
        ai_model_id="text-embedding-ada-002"
    )
    kernel.add_service(embedding_gen)

    # Setup memory using VolatileMemoryStore
    memory_store = VolatileMemoryStore()
    memory = SemanticTextMemory(storage=memory_store, embeddings_generator=embedding_gen)
    return kernel, memory

# %%
# Initialize kernel and memory
async def init():
    return await setup_kernel_and_memory()

kernel, memory = asyncio.run(init())

# %% [markdown]
# ## Define Helper Functions
# 
# Create functions to fetch papers and store them in memory:

# %%
async def fetch_papers(query: str, max_results: int = 5) -> List[arxiv.Result]:
    """Fetch papers from arXiv based on query."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    return list(search.results())

async def store_papers(memory: SemanticTextMemory, papers: List[arxiv.Result]):
    """Store paper abstracts in semantic memory."""
    for paper in papers:
        await memory.save_information(
            collection="papers",
            text=paper.summary,
            description=paper.title,
            additional_metadata={
                "authors": ", ".join(author.name for author in paper.authors),
                "url": paper.entry_id
            }
        )

async def generate_summary(kernel: Kernel, memory: SemanticTextMemory, topic: str) -> str:
    """Generate a summary of the papers in memory."""
    summarize = KernelFunctionFromPrompt(
        function_name="summarize_research",
        plugin_name="ResearchPlugin",
        description="Summarizes research papers on a specific topic",
        prompt="""
        Research Topic: {{$topic}}
        Papers to analyze:
        {{$papers}}

        Create a comprehensive research summary that:
        1. Identifies the main themes and findings
        2. Highlights significant breakthroughs
        3. Notes areas of consensus and disagreement
        4. Suggests future research directions

        Format the summary in a clear, academic style.
        """
    )

    # Get papers from memory
    papers = await memory.search("papers", topic, limit=5)
    papers_text = "\n\n".join([
        f"Title: {paper.description}\nAbstract: {paper.text}"
        for paper in papers
    ])

    # Generate summary
    result = await kernel.invoke(
        summarize,
        topic=topic,
        papers=papers_text
    )
    return str(result)

# %% [markdown]
# ## Execute Research Pipeline
# 
# Now let's run our research pipeline with a specific topic:

# %%
async def run_research(topic: str, max_papers: int = 5):
    print(f"🔬 Fetching papers on: {topic}")
    papers = await fetch_papers(topic, max_papers)

    print(f"📚 Storing {len(papers)} papers in memory")
    await store_papers(memory, papers)

    print("\n🤖 Generating research summary...")
    summary = await generate_summary(kernel, memory, topic)

    print("\n📝 Research Summary:\n")
    print(summary)

# Run the research pipeline
topic = "quantum computing error correction"
asyncio.run(run_research(topic))

# %% [markdown]
# ## Try Different Topics
# 
# You can modify the topic and run the research pipeline again to explore different areas:
# 
# ```python
# # Example: Research a different topic
# topic = "large language models"
# asyncio.run(run_research(topic))
