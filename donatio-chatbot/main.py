from langchain_groq import ChatGroq  
from dotenv import load_dotenv 
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict 
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from langgraph.types import Command
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn 
import os

load_dotenv()
model = ChatGroq(model_name="llama-3.3-70b-versatile")

db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))

class State(TypedDict):
    query: str 
    question: str 
    answer: str 
    result: str
    sender_role: str
    
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]
    

def supervisor(state: State) -> Command:
    if state["sender_role"] == "user":
        return Command(goto="user_agent", update={})
    elif state["sender_role"] == "org":
        return Command(goto="org_agent", update={})
    return Command(goto=END, update={})

user_query_prompt = PromptTemplate.from_template(
    """
You are a SQL assistant for an application with user-specific data.
Generate a SQL query based on the user's question below.

Dialect: {dialect}
Top K: {top_k}
Schema:
{table_info}

User Question:
{question}

SQL Query:
"""
)

org_query_prompt = PromptTemplate.from_template(
    """
You are a SQL assistant for an application managing organization-level data.
Generate a SQL query from the organization's perspective.

Dialect: {dialect}
Top K: {top_k}
Schema:
{table_info}

Organization Question:
{question}

SQL Query:
"""
)

def write_user_query(state: State) -> State:
    """Generate a SQL query based on user's question."""
    prompt_template = user_query_prompt  
    prompt_str = prompt_template.format(
        dialect=db.dialect,
        top_k=6,
        table_info=db.get_table_info(),
        question=state["question"]
    )

    structured_llm = model.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt_str)

    return {"query": result["query"]}

def write_org_query(state: State) -> State:
    """Generate a SQL query based on organization's question."""
    prompt_template = org_query_prompt  
    prompt_str = prompt_template.format(
        dialect=db.dialect,
        top_k=6,
        table_info=db.get_table_info(),
        question=state["question"]
    )

    # Structured output parsing
    structured_llm = model.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt_str)

    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def sql_generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, REMEMBER Dont say about SQL Backend"
        "You are a helpful, friendly assistant who responds naturally and clearly. "
        "You adapt your tone to the user's input: summarize lists, provide explanations, or fulfill requests. "
        "Avoid sounding robotic or overly formal. Use plain language. "
        "**Do not ask any follow-up questions.** "
        "Just provide the most relevant, human-like response to the user's input."
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = model.invoke(prompt)
    return {"answer": response.content}

memory = MemorySaver()

builder = StateGraph(State)

builder.add_node("supervisor", supervisor)
builder.add_node("write_user_query", write_user_query)
builder.add_node("write_org_query", write_org_query)
builder.add_node("execute_query", execute_query)
builder.add_node("sql_generate_answer", sql_generate_answer)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda s: "write_user_query" if s["sender_role"] == "user" else "write_org_query",
    {"write_user_query": "write_user_query", "write_org_query": "write_org_query"}
)

builder.add_edge("write_user_query", "execute_query")
builder.add_edge("write_org_query", "execute_query")
builder.add_edge("execute_query", "sql_generate_answer")
builder.add_edge("sql_generate_answer", END)

graph = builder.compile(checkpointer=memory)


graph = builder.compile(checkpointer=memory)

app = FastAPI(name="SQL_chatbot")

class ModelGenerate(BaseModel):
    sender_role: str 
    question: str 
    
@app.post("/generate")
def generate(model_generate: ModelGenerate):
    config = {"thread_id":1}
    result = graph.invoke({"sender_role": model_generate.sender_role,"question": model_generate.question},config=config)
    return result['answer']

if __name__ == "__main__":
    uvicorn.run(app)