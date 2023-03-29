#!/usr/bin/env python3

import os
import sys

from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
if serpapi_api_key is None:
    print("Please set the SERPAPI_API_KEY environment variable.")
    sys.exit(1)

llm = OpenAI(temperature=0.9)
tools = load_tools(["serpapi"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

question = " ".join(sys.argv[1:])
agent.run(question)
