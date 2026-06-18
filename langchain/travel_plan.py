import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel

# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

# Define prompt template
travel_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("human", "{trip}")
])

# Model (using API key from env)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

# Parsers
text_parser = StrOutputParser()

# RunnableSequence
sequence_chain = RunnableSequence(first=travel_prompt, middle=[llm], last=text_parser)

# RunnablePassthrough
passthrough = RunnablePassthrough()

# RunnableParallel
parallel_chain = RunnableParallel({
    "flights": travel_prompt | llm | text_parser,
    "hotels": travel_prompt | llm | text_parser,
    "weather": travel_prompt | llm | text_parser,
    "raw_input": passthrough
})

# Run
result = parallel_chain.invoke({"trip": "Plan my trip to Paris in July 2026"})
print(result)
