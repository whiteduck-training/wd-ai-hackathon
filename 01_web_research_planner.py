import asyncio
import json
import requests
from typing import Annotated
from bs4 import BeautifulSoup

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.planners import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)
from semantic_kernel.functions import kernel_function, KernelFunctionFromPrompt


class WebResearchPlugin:
    """Plugin for web research capabilities"""

    @kernel_function(
        name="SearchWeb",
        description="Searches the web using DuckDuckGo and returns relevant URLs",
    )
    def search_web(
        self, query: Annotated[str, "the search query to find relevant web pages"]
    ) -> str:
        # Using DuckDuckGo API (no key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url)
        data = json.loads(response.text)

        # Extract top results
        results = []
        for result in data.get("Results", [])[:3]:
            results.append(result["FirstURL"])

        return json.dumps(results)

    @kernel_function(
        name="ExtractContent", description="Extracts main content from a webpage"
    )
    def extract_content(
        self, url: Annotated[str, "the webpage URL to extract content from"]
    ) -> str:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, and navigation
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Extract text
            text = soup.get_text(separator=" ", strip=True)
            # Limit text length
            return text[:2000] + "..." if len(text) > 2000 else text
        except Exception as e:
            return f"Error extracting content: {str(e)}"


class ResearchPlugin:
    """Plugin for analyzing and summarizing research"""

    def __init__(self, memory: SemanticTextMemory):
        self.memory = memory

    @kernel_function(
        name="SaveToMemory", description="Saves research information to semantic memory"
    )
    async def save_to_memory(
        self,
        content: Annotated[str, "the content to save to memory"],
        topic: Annotated[str, "the research topic for categorization"],
    ) -> str:
        try:
            # Save to memory
            await self.memory.save_information(
                collection="research_data",
                text=content,
                description=f"Research on {topic}",
                additional_metadata={"topic": topic},
            )
            return "Content saved to memory successfully"
        except Exception as e:
            return f"Error saving to memory: {str(e)}"


async def setup_kernel_and_memory():
    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI service
    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
        ),
    )

    # Initialize memory (you'll need to configure this based on your setup)
    memory = SemanticTextMemory()

    # Add our research plugins
    kernel.add_plugin(WebResearchPlugin(), "web")
    kernel.add_plugin(ResearchPlugin(memory), "research")

    # Create semantic functions for analysis
    analyze_function = KernelFunctionFromPrompt(
        function_name="AnalyzeContent",
        plugin_name="ResearchPlugin",
        prompt="""
        Content: {{$input}}
        
        Analyze this content and extract key points. Focus on:
        1. Main concepts and ideas
        2. Key findings or statements
        3. Important relationships
        4. Credibility of information
        
        Format your response as bullet points.
        """,
        description="Analyzes and extracts key points from content.",
    )

    summarize_function = KernelFunctionFromPrompt(
        function_name="CreateSummary",
        plugin_name="ResearchPlugin",
        prompt="""
        Research Topic: {{$topic}}
        Collected Information: {{recall 'research_data' topic}}
        
        Create a comprehensive summary that:
        1. Synthesizes the main findings
        2. Highlights key agreements and contradictions
        3. Identifies gaps in the information
        4. Suggests areas for further research
        
        Keep the summary clear and well-structured.
        """,
        description="Creates a summary from collected research.",
    )

    kernel.add_function(plugin_name="ResearchPlugin", function=analyze_function)
    kernel.add_function(plugin_name="ResearchPlugin", function=summarize_function)

    return kernel


async def conduct_research(kernel: Kernel, task: str):
    # Create the stepwise planner
    planner = FunctionCallingStepwisePlanner(
        service_id="default",
        options=FunctionCallingStepwisePlannerOptions(
            max_iterations=15,
            max_tokens=4000,
        ),
    )

    try:
        print(f"\nResearch Task: {task}\n")
        print("Starting research process...")

        result = await planner.invoke(kernel, task)

        print("\nResearch Results:")
        print(result.final_answer)

        print("\nThought Process:")
        for thought in result.chat_history:
            print(f"- {thought}")

    except Exception as e:
        print(f"Error during research: {str(e)}")


async def main():
    # Example research tasks
    research_tasks = [
        """
        Research the latest developments in quantum computing.
        Focus on recent breakthroughs in error correction.
        Analyze the information and create a summary.
        """,
        """
        Find information about artificial intelligence in healthcare.
        Focus on recent applications in diagnostic imaging.
        Create a summary of the findings.
        """,
    ]

    print("🔍 Web Research Assistant with Semantic Kernel 🔍\n")

    # Setup kernel and memory
    kernel = await setup_kernel_and_memory()

    # Process each research task
    for task in research_tasks:
        await conduct_research(kernel, task)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
