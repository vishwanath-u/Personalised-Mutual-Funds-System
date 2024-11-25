import pandas as pd
import chainlit as cl
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.model_selection import cross_validate


# Global variables to store user input and conversation state
investment_amount = None
age = None
goal = None
strategy = None
state = 'start'  # Tracks which part of the conversation we're in

# 1. Load and Prepare Data
def load_and_prepare_data(file_path):
    # Load the data and handle any missing values
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['scheme_name', 'category', 'min_sip', 'min_lumpsum', 'rating'])

    reader = Reader(rating_scale=(1, 5))
    ratings_data = data[['scheme_name', 'category', 'min_sip', 'min_lumpsum', 'rating']].copy()
    ratings_data['user_id'] = [i % 10 for i in range(len(ratings_data))]  # Simulate user IDs
    ratings_data = ratings_data.rename(columns={'scheme_name': 'item_id', 'rating': 'rating'})
    surprise_data = Dataset.load_from_df(ratings_data[['user_id', 'item_id', 'rating']], reader)
    return data, surprise_data

# 2. Model Training
def train_model(surprise_data):
    trainset, testset = train_test_split(surprise_data, test_size=0.25)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    cross_val_results = cross_validate(model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return model, cross_val_results

# 3. Scheme Recommendation
def recommend_schemes(investment_amount, category_name=None, top_n=5, data=None):
    filtered_data = data[(data['min_sip'] <= investment_amount) | (data['min_lumpsum'] <= investment_amount)]
    if category_name:
        filtered_data = filtered_data[filtered_data['category'] == category_name]
    top_recommendations = filtered_data.sort_values(by='rating', ascending=False).head(top_n)
    return top_recommendations



# Define the determine_category function
def determine_category(age, goal, strategy):
    # Implement logic to determine the category based on age, goal, and strategy
    if age < 30 and goal == "wealth creation" and strategy == "aggressive":
        return "Equity"
    elif age > 50 and goal == "retirement" and strategy == "conservative":
        return "Debt"
    return "Balanced"  # Default fallback category

# Define the main function
@cl.on_message
async def main(message: cl.Message):
    global investment_amount, age, goal, strategy, state

    # Start the conversation by asking for the investment amount
    if state == 'start':
        await cl.Message(content="Welcome to the Mutual Fund Recommender! How much do you plan to invest?").send()
        state = 'waiting_for_investment'

    # Capture investment amount and ask for age
    elif state == 'waiting_for_investment':
        if message.content.isdigit():
            investment_amount = int(message.content)
            await cl.Message(content="What is your age?").send()
            state = 'waiting_for_age'
        else:
            await cl.Message(content="Please enter a valid investment amount (a number).").send()

    # Capture age and ask for investment goal
    elif state == 'waiting_for_age':
        if message.content.isdigit():
            age = int(message.content)
            await cl.Message(content="What is your investment goal? (e.g., retirement, wealth creation, etc.)").send()
            state = 'waiting_for_goal'
        else:
            await cl.Message(content="Please enter a valid age (a number).").send()

    # Capture investment goal and ask for strategy
    elif state == 'waiting_for_goal':
        goal = message.content
        await cl.Message(content="What is your investment strategy? (e.g., conservative, moderate, aggressive)").send()
        state = 'waiting_for_strategy'

    # Capture strategy and give recommendations
    elif state == 'waiting_for_strategy':
        strategy = message.content
        # Determine the category based on the age, goal, and strategy
        category_name = determine_category(age, goal, strategy)

        # Load the data to recommend schemes
        file_path = '/Users/vichu/DataspellProjects/contentRS/data.xlsx'
        data, _ = load_and_prepare_data(file_path)

        # Get top schemes based on the user input
        top_schemes = recommend_schemes(investment_amount, category_name, top_n=5, data=data)
        scheme_message = top_schemes[['scheme_name', 'category', 'min_sip', 'min_lumpsum', 'rating']].to_string(index=False)

        # Send the top recommendations to the user
        await cl.Message(content=f"Here are the top recommended schemes in the {category_name} category:\n\n{scheme_message}").send()


        # Reset state to start a new conversation if desired
        state = 'start'

if __name__ == '__main__':
    cl.run()