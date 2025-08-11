import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.googlesearch import GoogleSearchTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Define Agents ---
web_agent = Agent(
    name="Web Agent",
    role="search the web for information",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[DuckDuckGoTools()],
    instructions="Always include the sources",
    show_tool_calls=True,
    markdown=True,
)
google_agent = Agent(
    name="G Agent",
    tools=[GoogleSearchTools()],
    description="You are a news agent that helps users find the latest news.",
    model=Groq(id="qwen/qwen3-32b"),
    instructions=[
        "Given a topic by the user, respond with 4 latest news items about that topic.",
        "Search for 10 news items and select the top 4 unique items.",
        "Search in English and in French.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True
        )
    ],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[google_agent, finance_agent],
    model=Groq(id="qwen/qwen3-32b"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit UI ---
st.set_page_config(page_title="Finance & Web Agent", page_icon="💹")

st.title("💹 Multi-Agent Financial & Web Analysis Tool")
st.write("Ask me about companies, markets, or any other financial info.")

# Sidebar to choose agent
st.sidebar.header("Agent Selection")
agent_choice = st.sidebar.radio(
    "Choose which agent to use:",
    ("Web Agent", "Finance Agent", "Both (Team)")
)

# Map choice to agent instance
if agent_choice == "Web Agent":
    selected_agent = google_agent
elif agent_choice == "Finance Agent":
    selected_agent = finance_agent
else:
    selected_agent = agent_team

# User input
user_query = st.text_area(
    "Enter your query:",
    placeholder="Example: Analyze Tesla, NVDA, and Apple for long-term investment"
)

# Run analysis
if st.button("Run Analysis"):
    if user_query.strip():
        with st.spinner(f"Running {agent_choice}... please wait"):
            try:
                result = selected_agent.run(user_query)

                # Ensure structured clean output
                if result and hasattr(result, "content"):
                    st.markdown(result.content)  # Only display main answer
                else:
                    st.warning("No content returned from the agent.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query first.")