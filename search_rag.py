import os
import time
import warnings
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily.errors import InvalidAPIKeyError
import chainlit as cl
from langchain.schema.runnable import Runnable
from typing import cast

warnings.filterwarnings('ignore')

sent_messages = []

def stream_output(response: str):
    for i in response.split():
        yield i + " "
        time.sleep(0.1)

def search_web(query: str) -> list:
    """Search the web."""
    try:
        tavily_tool = TavilySearchAPIRetriever(
            k=3,
            api_key=os.getenv("TAVILY_API_KEY"),
            max_tokens=10000,
            search_depth='advanced'
        )
        results = tavily_tool.invoke(query)
        if not results:
            return ["No relevant context found."]
        return [result.page_content for result in results]
    except InvalidAPIKeyError as e:
        return ["Invalid API key."]
    except Exception as e:
        return [f"Error occurred while fetching context: {str(e)}"]

@cl.on_chat_start
async def on_query():
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not groq_key or not tavily_key:
        await cl.Message(content="API keys are missing. Please provide valid keys.").send()
        return

    llm = ChatGroq(
        api_key=groq_key,
        model='llama-3.2-1b-preview',
        temperature=0,
        max_retries=3,
        timeout=None,
        max_tokens=1024
    )

    prompt = """You are a helpful assistant. Here is the conversation history so far:
            {conversation_history}
          
            Answer the question according to the query and given context:
            Question: {question}
            Context: {context}
            Provide an accurate response in bullet points but don't mention it in the response,
            the answer should be brief (max 5 lines/points).
            Do not hallucinate.
    """

    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = prompt_template | llm

    cl.user_session.set("runnable", chain)
    cl.user_session.set("conversation_history", "")

@cl.on_message
async def main(query: cl.Message):

    if query.content.lower() == "clear":
        cl.user_session.set("conversation_history", "")
        
        for msg in sent_messages:
            await msg.remove() 
        
        sent_messages.clear()

        await cl.Message(content="Chat cleared. Start a new conversation!").send()
        return

    runnable = cast(Runnable, cl.user_session.get("runnable"))
    
    conversation_history = cl.user_session.get("conversation_history", "")
    
    context = search_web(query.content)

    response = runnable.invoke(
        {
            "question": query.content,
            "context": context,
            "conversation_history": conversation_history
        }
    )
    
    conversation_history += f"User: {query.content}\nAI: {response.content}\n"
    
    cl.user_session.set("conversation_history", conversation_history)
    
    ai_msg = await cl.Message(content=f"AI: {response.content}").send()

    sent_messages.append(query)
    sent_messages.append(ai_msg)
