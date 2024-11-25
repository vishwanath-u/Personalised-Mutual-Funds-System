import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


@cl.on_message
async def handle_message(message):
    global user_session

    if user_session["state"] == 'start':
        await cl.Message(content="Welcome to the Mutual Fund Recommender! How much do you plan to invest?").send()
        user_session["state"] = 'waiting_for_investment'

    elif user_session["state"] == 'waiting_for_investment':
        if message.content.isdigit():
            user_session["investment_amount"] = int(message.content)
            await cl.Message(content="What is your age?").send()
            user_session["state"] = 'waiting_for_age'
        else:
            await cl.Message(content="Please enter a valid investment amount (a number).").send()

    elif user_session["state"] == 'waiting_for_age':
        if message.content.isdigit():
            user_session["age"] = int(message.content)
            await cl.Message(content="What is your investment goal? (e.g., retirement, wealth creation, etc.)").send()
            user_session["state"] = 'waiting_for_goal'
        else:
            await cl.Message(content="Please enter a valid age (a number).").send()

    elif user_session["state"] == 'waiting_for_goal':
        investing_goal = message.content
        user_session["investing_goal"] = investing_goal

        # Retrieve all embedded documents
        retrieved_docs = retriever.get_relevant_documents("")  # Empty string to get all documents
        context = "\n".join(doc.page_content for doc in retrieved_docs)

        # Generate portfolio strategy
        investment_amount = user_session["investment_amount"]
        age = user_session["age"]

        # Define the system prompt with context as an input variable
        system_prompt = (
            "You are a portfolio advisor assistant. "
            "Use the following pieces of retrieved context to provide "
            "a personalized portfolio strategy based on the user's "
            "input. Recommend a mix of funds that align with the user's "
            "investment goal and allocate the investment amount of ₹{investment_amount:.2f} "
            "across the recommended funds. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "User's age: {age}\n"
            "User's investment amount: ${investment_amount:.2f}\n"
            "User's investment goal: {investing_goal}\n"
            "Retrieved context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Prepare input data
        input_data = {
            "input": message.content,  # User's question
            "age": user_session["age"],
            "investment_amount": user_session["investment_amount"],
            "investing_goal": user_session["investing_goal"],
            "context": context
        }

        # Get the output directly instead of streaming
        response = rag_chain.invoke(input_data)  # Use .run() to get the output
        answer = response.get('answer', 'Sorry, I couldn’t find an answer to your query.')

        # Send the complete response to the user
        await cl.Message(content=answer).send()

        user_session["state"] = 'done'


# Run the Chainlit app
if __name__ == '__main__':
    cl.run()
