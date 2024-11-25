import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser
from pydantic import BaseModel, Field
import asyncio

# Step 1: Load, chunk, and index the contents of the CSV file
loader = CSVLoader('/Users/vichu/Downloads/comprehensive_mutual_funds_data.csv')
docs = loader.load()  # Load the CSV file

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create a vector store for document embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Initialize the LLM
llm = ChatOpenAI(temperature=0.2, model='gpt-3.5-turbo')

# Define state management
user_session = {"state": "start"}

# Define a Pydantic model for the expected output
class SummaryResponse(BaseModel):
    output: str = Field(..., description="The investment strategy summary.")

def devise_strategy(age, investment_goal):
    if age < 30:
        return "aggressive" if investment_goal == "wealth creation" else "moderate"
    elif age < 50:
        return "moderate" if investment_goal == "wealth creation" else "conservative"
    else:
        return "conservative"

def recommend_funds(investment_amount, goal, age, funds):
    strategy = devise_strategy(age, goal)
    recommended_funds = []

    for fund in funds:
        if strategy == "aggressive" and fund['risk_level'] <= 3:
            recommended_funds.append(fund)
        elif strategy == "moderate" and fund['risk_level'] <= 5:
            recommended_funds.append(fund)
        elif strategy == "conservative" and fund['risk_level'] <= 7:
            recommended_funds.append(fund)

    num_funds = len(recommended_funds)
    allocation = investment_amount / num_funds if num_funds > 0 else 0

    recommendations = []
    for fund in recommended_funds:
        recommendations.append({
            "fund_name": fund['scheme_name'],
            "allocation": allocation,
            "returns_3yr": fund['returns_3yr'],
            "risk_level": fund['risk_level'],
            "rating": fund['rating']
        })

    return recommendations

# Define the investment strategy prompt template
prompt_template = PromptTemplate(
    input_variables=["investment_goal", "age", "investment_amount", "recommendations"],
    template="""
User's Investment Goal:
{investment_goal}

User's Age:
{age}

User's Investment Amount:
{investment_amount}

Recommended Funds:
{recommendations}

Based on the information provided above, generate a concise investment strategy summary that highlights the rationale for the recommended funds and how they align with the user's goals.
"""
)

@cl.on_message
async def handle_message(message):
    global user_session

    if user_session["state"] == 'start':
        asyncio.create_task(cl.Message(content="Welcome to the Mutual Fund Recommender! How much do you plan to invest?").send())
        user_session["state"] = 'waiting_for_investment'

    elif user_session["state"] == 'waiting_for_investment':
        if message.content.isdigit():
            user_session["investment_amount"] = int(message.content)
            asyncio.create_task(cl.Message(content="What is your age?").send())
            user_session["state"] = 'waiting_for_age'
        else:
            asyncio.create_task(cl.Message(content="Please enter a valid investment amount (a number).").send())

    elif user_session["state"] == 'waiting_for_age':
        if message.content.isdigit():
            user_session["age"] = int(message.content)
            asyncio.create_task(cl.Message(content="What is your investment goal? (e.g., retirement, wealth creation, etc.)").send())
            user_session["state"] = 'waiting_for_goal'
        else:
            asyncio.create_task(cl.Message(content="Please enter a valid age (a number).").send())

    elif user_session["state"] == 'waiting_for_goal':
        investing_goal = message.content
        user_session["investing_goal"] = investing_goal

        # Retrieve all embedded documents
        retrieved_docs = retriever.get_relevant_documents("")  # Empty string to get all documents
        context = [doc.page_content for doc in retrieved_docs]

        # Assuming each line of context is a fund's details in a dictionary-like format
        funds = []
        for line in context:
            fund_info = {}
            for item in line.split('\n'):
                if ': ' in item:  # Ensure there's a colon in the line
                    key, value = item.split(': ', 1)  # Split on the first occurrence
                    fund_info[key.strip()] = value.strip()
            if fund_info:  # Only add non-empty fund_info
                funds.append(fund_info)

        # Convert numeric fields for processing
        for fund in funds:
            fund['risk_level'] = int(fund['risk_level'])
            fund['returns_3yr'] = float(fund['returns_3yr'])

        # Get fund recommendations based on devised strategy
        recommendations = recommend_funds(user_session["investment_amount"], investing_goal, user_session["age"], funds)

        # Prepare the recommendation text
        recommendation_text = "\n".join(
            [
                f"Invest ${rec['allocation']:.2f} in {rec['fund_name']} (Risk Level: {rec['risk_level']}, Rating: {rec['rating']}). Expected returns over 3 years: {rec['returns_3yr']}%."
                for rec in recommendations]
        )

        # Prepare the input for the prompt template
        input_data = {
            "investment_goal": user_session["investing_goal"],
            "age": user_session["age"],
            "investment_amount": user_session["investment_amount"],
            "recommendations": recommendation_text
        }

        # Generate response using the LLM with structured output parser
        parser = StructuredOutputParser(response_schema=SummaryResponse)
        llm_response = (llm | prompt_template | parser).invoke(input_data)

        # Extract the summary from the LLM response
        summary = llm_response.output if llm_response else 'Unable to generate a summary.'

        asyncio.create_task(cl.Message(content=f"Here is the investment strategy summary:\n{summary}").send())
        user_session["state"] = 'done'

# Run the Chainlit app
if __name__ == '__main__':
    cl.run()
