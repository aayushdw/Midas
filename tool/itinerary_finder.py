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
    num_itinerary: int
    itineraries: str

def find_itineraries(location: str, num_itinerary: int) -> str:
    """Finds existing itineraries for a given location."""
    llm = get_llm()

    def find_itineraries_node(state: GraphState) -> dict[str, str]:
        prompt = f"Find touristy activities of things to do in '{state['location']}'. I want an itemized list only."
        print(prompt)
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return {"itineraries": str(response.content)}

    workflow = StateGraph(GraphState)
    workflow.add_node("find_itineraries_node", find_itineraries_node)
    workflow.set_entry_point("find_itineraries_node")
    workflow.add_edge("find_itineraries_node", END)
    app = workflow.compile()

    final_state = app.invoke(cast(GraphState, {"location": location, "num_itinerary": num_itinerary}))
    return final_state['itineraries']

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
        parser = argparse.ArgumentParser(description="Find existing itineraries for a given location.")
        parser.add_argument("location", type=str, help="The location to find itineraries for.")
        parser.add_argument("--num_itinerary", type=int, default=10, help="The number of itineraries to find.")
        args = parser.parse_args()

        itineraries = find_itineraries(args.location, args.num_itinerary)
        formatted_itineraries = format_links(itineraries)
        print(formatted_itineraries)
