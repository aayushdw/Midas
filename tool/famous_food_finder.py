import argparse
import os
import re
from dotenv import load_dotenv

# Load the .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=dotenv_path)

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, cast
from llm_provider import get_llm

class GraphState(TypedDict):
    location: str
    famous_foods: str

def find_famous_foods(location: str) -> str:
    """Finds famous foods for a given location."""
    llm = get_llm()

    def find_famous_foods_node(state: GraphState) -> dict[str, str]:
        prompt = f"Find famous foods in '{state['location']}'. I want an itemized list only."
        print(prompt)
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return {"famous_foods": str(response.content)}

    workflow = StateGraph(GraphState)
    workflow.add_node("find_famous_foods_node", find_famous_foods_node)
    workflow.set_entry_point("find_famous_foods_node")
    workflow.add_edge("find_famous_foods_node", END)
    app = workflow.compile()

    final_state = app.invoke(cast(GraphState, {"location": location}))
    return final_state['famous_foods']

def format_links(links: str) -> str:
    """Formats the links to be green and underlined."""
    # Regular expression to find URLs
    url_regex = r'(https?://[^\s]+)'
    
    def replace_url_with_color(match: re.Match[str]) -> str:
        return f"\033[92;4m{match.group(0)}\033[0m"
    
    return re.sub(url_regex, replace_url_with_color, links)

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in .env file.")
    else:
        parser = argparse.ArgumentParser(description="Find famous foods for a given location.")
        parser.add_argument("location", type=str, help="The location to find famous foods for.")
        args = parser.parse_args()

        famous_foods = find_famous_foods(args.location)
        formatted_famous_foods = format_links(famous_foods)
        print(formatted_famous_foods)
