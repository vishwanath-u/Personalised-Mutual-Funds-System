import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import cast

# Load the dataset and create the vector database
loader = CSVLoader('/Users/vichu/Downloads/comprehensive_mutual_funds_data.csv')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(documents, embeddings)
retriever = vector_db.as_retriever()

# Define the Chainlit app
class PortfolioAdvisor(cl.ChainlitApp):
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.2, model='gpt-3.5-turbo', streaming=True)
        self.state = 'start'

    async def message_handler(self, message):
        if self.state == 'start':
            await cl.Message(content="Welcome to the Mutual Fund Recommender! How much do you plan to invest?").send()
            self.state = 'waiting_for_investment'
        elif self.state == 'waiting_for_investment':
            if message.content.isdigit():
                cl.user_session.set("investment_amount", int(message.content))
                await cl.Message(content="What is your age?").send()
                self.state = 'waiting_for_age'
            else:
                await cl.Message(content="Please enter a valid investment amount (a number).").send()
        elif self.state == 'waiting_for_age':
            if message.content.isdigit():
                cl.user_session.set("age", int(message.content))
                await cl.Message(content="What is your investment goal? (e.g., retirement, wealth creation, etc.)").send()
                self.state = 'waiting_for_goal'
            else:
                await cl.Message(content="Please enter a valid age (a number).").send()
        elif self.state == 'waiting_for_goal':
            investing_goal = message.content
            cl.user_session.set("investing_goal", investing_goal)

            # Retrieve context for personalized strategy
            investment_amount = cl.user_session.get("investment_amount")
            age = cl.user_session.get("age")
            query = f"Investment goal: {investing_goal}, Age: {age}, Amount: {investment_amount}"

            retrieved_docs = retriever.retrieve(query)
            retrieved_context = "\n".join(doc.content for doc in retrieved_docs)

            # Generate portfolio strategy
            system_prompt = (
                "You are a portfolio advisor assistant. "
                "Use the following pieces of retrieved context to provide "
                "a personalized portfolio strategy based on the user's "
                "input. Recommend a mix of funds that align with the user's "
                "investment goal and allocate the investment amount of ${investment_amount:.2f} "
                "across the recommended funds. If you don't know the answer, say that you don't know. "
                "Use three sentences maximum and keep the answer concise."
                "\n\n"
                "User's age: {age}\n"
                "User's investment amount: ${investment_amount:.2f}\n"
                "User's investment goal: {investing_goal}\n"
                "Retrieved context from the vector database: {retrieved_context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt.format(
                        age=age,
                        investment_amount=investment_amount,
                        investing_goal=investing_goal,
                        retrieved_context=retrieved_context
                    )),
                    ("human", ""),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            async for chunk in rag_chain.astream({"input": ""}, config=None):
                await message.stream_token(chunk)

            await message.send()
            self.state = 'done'

# Create and run the Chainlit app
if __name__ == '__main__':
    cl.run(PortfolioAdvisor)
