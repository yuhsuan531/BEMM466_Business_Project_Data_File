import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from wordcloud import WordCloud
import os

# Get the directory where the current script is located
current_dir = os.path.dirname(__file__)

# Construct the relative path to the CSV file
file_path = os.path.join(current_dir, 'BEMM466_Business_Project_Data_File', 'Cleaned_Trustpilot_Reviews.csv')

# Load the CSV file using the relative path
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Display the first few rows of the dataframe
print(df.head())

# Calculate review length (number of words)
df['review_length_words'] = df['Review'].apply(lambda x: len(x.split()))

# Convert the review date to datetime and add a column for the day of the week
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Day_of_Week'] = df['Date'].dt.day_name()

# Set consistent color palette
sns.set_palette("viridis")

# 1. Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='Rating', data=df)
plt.title('Distribution of Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# 2. Total Rating Change by Day of the Week
ratings_by_day = df.groupby('Day_of_Week').size().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Print out the result for each day
print("Total Ratings by Day of the Week:")
print(ratings_by_day)

# Plot: Total Rating Change by Day of the Week
plt.figure(figsize=(10, 6))
sns.lineplot(x=ratings_by_day.index, y=ratings_by_day.values, marker='o')
plt.title('Total Rating Change by Day of the Week', fontsize=16)
plt.xlabel('')
plt.ylabel('Total Number of Ratings', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Each Rating Change by Day of the Week
rating_by_day = df.groupby(['Day_of_Week', 'Rating']).size().unstack(fill_value=0)
rating_by_day = rating_by_day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Define a color palette for the ratings (5 distinct colors)
color_palette = sns.color_palette(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Plot: Each Rating Change by Day of the Week
plt.figure(figsize=(12, 8))
sns.lineplot(data=rating_by_day, markers=True, dashes=False, marker='o', markersize=10)
plt.title('Each Rating Change by Day of the Week', fontsize=20)
plt.xlabel('')
plt.ylabel('Number of Ratings', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title='Rating', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Average Rating Trend Over Time
# Extract year and month for grouping
df['YearMonth'] = df['Date'].dt.to_period('M')

# Calculate the average rating for each month
average_rating_per_month = df.groupby('YearMonth')['Rating'].mean().reset_index()

# Plot the average rating trend over time
plt.figure(figsize=(12, 6))
plt.plot(average_rating_per_month['YearMonth'].astype(str), average_rating_per_month['Rating'], marker='o')
plt.title('Average Rating Trend Over Time', fontsize=18)
plt.xlabel('')
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45,fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Relationship Between Review Length and Rating
# Group by rating and calculate the average review length (in words)
avg_length_by_rating = df.groupby('Rating')['review_length_words'].mean().reset_index()

# Plotting the relationship between rating and average review length (in words)
plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='review_length_words', data=avg_length_by_rating, palette='viridis')

# Adding titles and labels
plt.title('Average Review Length by Rating', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Average Review Length (in words)', fontsize=14)

# Adding the average length numbers on top of each bar
for index, row in avg_length_by_rating.iterrows():
    plt.text(row['Rating'] - 1, row['review_length_words'] + 1, 
             f"{row['review_length_words']:.1f}", color='black', ha="center", fontsize=12)

# Setting the font size of the tick labels
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  

# Show the plot
plt.show()

# 6. The distribution of review lengths (in words) for each rating
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Rating', y='review_length_words', data=df, palette='viridis')

# Adding titles and labels
plt.title('Distribution of Review Lengths by Rating', fontsize=22)
plt.xlabel('Rating', fontsize=20)
plt.ylabel('Review Length (in words)', fontsize=20)

# Setting the font size of the tick labels
plt.xticks(fontsize=18)  
plt.yticks(fontsize=18)  

# Adding IQR annotations
summary_stats = df.groupby('Rating')['review_length_words'].describe()
for i, rating in enumerate(summary_stats.index):
    q1 = summary_stats.loc[rating, '25%']
    q3 = summary_stats.loc[rating, '75%']
    iqr = q3 - q1
    ax.text(i+0.03, q3 + 2.5, f'IQR: {iqr:.2f}', 
            horizontalalignment='center', fontsize=14, color='black')

# Show the plot
plt.show()

# Display summary statistics and export to CSV
summary_stats.to_csv(r"C:\Users\taylo\Desktop\business project\summary_statistics.csv")
print(f"Summary statistics have been exported to C:\\Users\\taylo\\Desktop\\business project\\summary_statistics.csv")

# Print the summary statistics for IQR reference
print(summary_stats[['25%', '75%']])

# 7. Distribution of Sentiment Categories
def categorize_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Review'].apply(categorize_sentiment)

sentiment_distribution = df['Sentiment'].value_counts(normalize=True) * 100
print("Sentiment Distribution:")
print(sentiment_distribution)

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values, palette='viridis')
plt.title('Distribution of Sentiment Categories by Review Text', fontsize=16)
plt.xlabel('Sentiment Category', fontsize=14)
plt.ylabel('Percentage of Reviews', fontsize=14)
plt.xticks(fontsize=14, rotation=0)

# Annotate the bars with the percentage values
for index, value in enumerate(sentiment_distribution):
    plt.text(index, value + 0.5, f'{value:.2f}%', ha='center')

plt.show()

# 8. Distribution of Sentiment Scores
# Perform sentiment analysis to calculate polarity for each review
df['Sentiment_Polarity'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Sentiment_Polarity'], kde=True, color=sns.color_palette("viridis")[0], bins=30)

plt.title('Distribution of Sentiment Scores', fontsize=18)
plt.xlabel('Sentiment Polarity', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)
plt.grid(True)
plt.show()

# Calculate summary statistics
min_score = df['Sentiment_Polarity'].min()
max_score = df['Sentiment_Polarity'].max()
mean_score = df['Sentiment_Polarity'].mean()

# Define ranges for sentiment scores
ranges = {
    'Very Negative (-1 to -0.5)': ((-1, -0.5), 0),
    'Negative (-0.5 to 0)': ((-0.5, 0), 0),
    'Neutral (0)': ((0, 0), 0),
    'Positive (0 to 0.5)': ((0, 0.5), 0),
    'Very Positive (0.5 to 1)': ((0.5, 1), 0)
}

# Calculate the percentage of sentiment scores within each range
total_reviews = len(df)
for range_name, ((low, high), count) in ranges.items():
    count = df[(df['Sentiment_Polarity'] > low) & (df['Sentiment_Polarity'] <= high)].count()['Sentiment_Polarity']
    percentage = (count / total_reviews) * 100
    ranges[range_name] = percentage

# Print summary
print(f"Summary of Sentiment Scores:")
print(f"Minimum Sentiment Score: {min_score}")
print(f"Maximum Sentiment Score: {max_score}")
print(f"Mean Sentiment Score: {mean_score:.2f}\n")

print("Percentage of Sentiment Scores in each range:")
for range_name, percentage in ranges.items():
    print(f"{range_name}: {percentage:.2f}%")

# 9. Topic Modelling of each Sentiment Category
# Define a function to categorize sentiment
def categorize_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

# Apply the function to create a new column for sentiment
df['Sentiment'] = df['Rating'].apply(categorize_sentiment)

# Separate the reviews by sentiment
positive_reviews = df[df['Sentiment'] == 'positive']['Review']
neutral_reviews = df[df['Sentiment'] == 'neutral']['Review']
negative_reviews = df[df['Sentiment'] == 'negative']['Review']

# Function to perform topic modeling
def perform_topic_modeling(reviews, num_topics=5, num_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    topics = {}
    for idx, topic in enumerate(lda.components_):
        topics[f'Topic {idx + 1}'] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
    
    return topics

# Perform topic modeling for each sentiment category
positive_topics = perform_topic_modeling(positive_reviews)
neutral_topics = perform_topic_modeling(neutral_reviews)
negative_topics = perform_topic_modeling(negative_reviews)

# Function to create a word cloud from topics
def create_wordcloud(topics, sentiment_label):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        {word: 1 for topic_words in topics.values() for word in topic_words})
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"{sentiment_label.capitalize()} Sentiment Topics", fontsize=16)
    plt.axis('off')
    plt.show()

# Generate word clouds for each sentiment
create_wordcloud(positive_topics, "positive")
create_wordcloud(neutral_topics, "neutral")
create_wordcloud(negative_topics, "negative")

# 10. Trend Of Review Sentiments Over Time
# Extract year and month for grouping
df['YearMonth'] = df['Date'].dt.to_period('M')

# Calculate the count of each sentiment per month
sentiment_trend = df.groupby(['YearMonth', 'Sentiment']).size().unstack().fillna(0)

# Export the sentiment trend data to an Excel file
output_file_path = r"C:\Users\taylo\Desktop\business project\sentiment_trend_over_time.xlsx"
sentiment_trend.to_excel(output_file_path)

# Define a color palette for the sentiments (3 distinct colors)
color_palette = sns.color_palette(['#1f77b4', '#ff7f0e', '#2ca02c'])

# Plot the sentiment trend over time
plt.figure(figsize=(14, 7))
plt.plot(sentiment_trend.index.astype(str), sentiment_trend['positive'], marker='o', label='Positive', color=color_palette[2])
plt.plot(sentiment_trend.index.astype(str), sentiment_trend['neutral'], marker='o', label='Neutral', color=color_palette[1])
plt.plot(sentiment_trend.index.astype(str), sentiment_trend['negative'], marker='o', label='Negative', color=color_palette[0])

# Highlight the different phases
plt.axvspan('2022-11', '2023-07', color='green', alpha=0.2, label='Growth Phase')
plt.axvspan('2023-08', '2024-02', color='red', alpha=0.2, label='Decline Phase')
plt.axvspan('2024-03', '2024-07', color='blue', alpha=0.2, label='Fluctuation Phase')

# Add titles and labels
plt.title('Trend of Review Sentiments Over Time', fontsize=20)
plt.xlabel('')
plt.ylabel('Count of Reviews', fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title='Sentiment', fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.show()

# 11. Correlation Between Sentiment, Reviews, App Downloads, and Days Since Start
# Load the app downloads data
downloads_data = pd.read_excel(r"C:\Users\taylo\Desktop\business project\data file\App_Downloads.xlsx")

# Merge with the app downloads data
downloads_data['Date'] = pd.to_datetime(downloads_data['Date'], format="%b '%y")
df = pd.merge(df, downloads_data, on='Date', how='inner')

# Calculate the number of days since the start
df['Days Since Start'] = (df['Date'] - df['Date'].min()).dt.days

# Calculate the volume of reviews (number of reviews per day)
df['Volume of Reviews'] = df.groupby('Date')['Review'].transform('count')

# Select the relevant columns for the pairplot and rename them
pairplot_data = df[['Sentiment_Polarity', 'Volume of Reviews', 'Number of App Downloads in millions', 'Days Since Start']].rename(
    columns={
        'Number of App Downloads in millions': 'App Downloads',
        'Sentiment_Polarity': 'Sentiment Polarity'  # Adjust the name to match the exact column in your DataFrame
    }
)
# Calculate the correlation matrix
correlation_matrix = pairplot_data.corr()

# Export the correlation matrix to an Excel file
output_file_path = r"C:\Users\taylo\Desktop\business project\correlation_coefficients.xlsx"
correlation_matrix.to_excel(output_file_path)

print(f"Correlation coefficients have been exported to {output_file_path}")

# Generate the pairplot with regression lines 
g=sns.pairplot(pairplot_data, kind="reg", plot_kws={'scatter_kws': {'s': 3}})

# Increase the size of x and y labels, and split y labels into two lines at a specific position
for ax in g.axes.flatten():
    # Insert a manual line break into the y-label text at a desired space
    y_label = ax.get_ylabel()
    if " " in y_label:
        parts = y_label.split(" ", 1)  # Split only at the first space
        y_label = f"{parts[0]}\n{parts[1]}"
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()  # Ensure everything fits well
plt.show()
