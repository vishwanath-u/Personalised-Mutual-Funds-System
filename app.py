import pandas as pd
import chainlit as cl
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
from io import BytesIO

# 1. Load and Prepare Data
def load_and_prepare_data(file_path):
    data = pd.read_excel(file_path)
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

# 4. Chart Generation for Asset Allocation
def generate_asset_allocation_chart(top_recommendations):
    """
    Generates a pie chart showing the asset allocation of the recommended funds.
    """
    allocation_data = top_recommendations[['scheme_name', 'equity_allocation', 'debt_allocation', 'cash_allocation']]

    # Generate a pie chart for each recommended scheme
    fig, axes = plt.subplots(1, len(allocation_data), figsize=(6 * len(allocation_data), 6))

    if len(allocation_data) == 1:
        axes = [axes]  # Wrap single axes to list

    for i, (_, row) in enumerate(allocation_data.iterrows()):
        labels = ['Equity', 'Debt', 'Cash']
        sizes = [row['equity_allocation'], row['debt_allocation'], row['cash_allocation']]
        axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[i].axis('equal')
        axes[i].set_title(row['scheme_name'])

    plt.tight_layout()

    # Save chart to a BytesIO buffer to send via Chainlit
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return img_buffer

# Define the determine_category function
def determine_category(age, goal, strategy):
    # Implement logic to determine the category based on age, goal, and strategy
    # For example:
    if age < 30 and goal == "wealth creation" and strategy == "aggressive":
        return "Equity"
    elif age > 50 and goal == "retirement" and strategy == "conservative":
        return "Debt"
    # ... and so on
    pass

# Define the main function
@cl.on_message
async def main(message: cl.Message, data: pd.DataFrame):
    global investment_amount, category_name, age, goal, strategy

    if message.content == "/start":
        await cl.Message(content="Welcome to the Mutual Fund Recommender! How much do you plan to invest?").send()

    elif message.content.isdigit():
        investment_amount = int(message.content)
        await cl.Message(content="What is your age?").send()

    elif message.content.isdigit() and age is not None:
        age = int(message.content)
        await cl.Message(content="What is your investment goal? (e.g. retirement, wealth creation, etc.)").send()

    elif message.content and age is not None and investment_amount is not None:
        goal = message.content
        await cl.Message(content="What is your investment strategy? (e.g. conservative, moderate, aggressive)").send()

    elif message.content and age is not None and investment_amount is not None and goal:
        strategy = message.content
        category_name = determine_category(age, goal, strategy)
        top_schemes = recommend_schemes(investment_amount, category_name, top_n=5, data=data)
        scheme_message = top_schemes[['scheme_name', 'category', 'min_sip', 'min_lumpsum', 'rating']].to_string(
            index=False)

        await cl.Message(
            content=f"Here are the top recommended schemes in the {category_name} category:\n\n{scheme_message}").send()

        # Generate and send asset allocation chart
        chart_img = generate_asset_allocation_chart(top_schemes)
        await cl.Image(content=chart_img.read(), name="Asset Allocation").send()

if __name__ == '__main__':
    file_path = '/Users/vichu/DataspellProjects/contentRS/data.xlsx'
    data, surprise_data = load_and_prepare_data(file_path)
    await main (data=data)
    model, cross_val_results = train_model(surprise_data)
    cl.run(data=data)