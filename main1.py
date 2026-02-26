from dotenv import load_dotenv

load_dotenv()
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama


llm = ChatOllama(model="gemma3:1b", temperature=0)
tavily_tool = TavilySearch(max_results=5)


def main():
    print("Hello from langchain-course!")
    query = "AI engineer LangChain Bay Area jobs"

    def build_prompt(inputs: dict) -> list[HumanMessage]:
        search_results = inputs["search_results"]
        user_query = inputs["query"]
        
        # Parse Tavily results và format gọn
        formatted_results = ""
        if isinstance(search_results, dict) and "results" in search_results:
            for idx, result in enumerate(search_results["results"], 1):
                title = result.get("title", "N/A")
                url = result.get("url", "N/A")
                content = result.get("content", "")[:200]  # Chỉ lấy 200 ký tự đầu
                formatted_results += f"{idx}. {title}\n   URL: {url}\n   Snippet: {content}\n\n"
        else:
            formatted_results = str(search_results)
        
        print("\n=== Formatted Results ===")
        print(formatted_results)
        print("=========================\n")
        
        return [
            HumanMessage(
                content=(
                    "Based on the job search results below, list the top 3 job postings with:\n"
                    "- Job Title\n"
                    "- URL\n"
                    "- Brief description\n\n"
                    f"Search results:\n{formatted_results}"
                )
            )
        ]

    pipeline = RunnableSequence(
        RunnableLambda(lambda q: {"query": q}),
        RunnableLambda(
            lambda inputs: {
                **inputs,
                "search_results": tavily_tool.invoke(
                    {"query": inputs["query"]},
                    config={"run_name": "tavily_search"},
                ),
            }
        ),
        RunnableLambda(build_prompt),
        llm,
    )

    result = pipeline.invoke(
        query,
        config={"run_name": "job_search_pipeline"},
    )
    print(result.content)


if __name__ == "__main__":
    main()
