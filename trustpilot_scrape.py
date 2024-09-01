import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import random

print("Starting script...")

# Base URL (without page number)
base_url = "https://uk.trustpilot.com/review/temu.com?sort=recency"

# Maximum number of pages to scrape (adjust as needed)
max_pages = 1000

# List of user agents to rotate
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36',
]

# Additional headers to make the request look more genuine
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# List to store reviews
reviews_list = []

# Function to fetch page content with retries and proxy support
def fetch_page(url, retries=3):
    for attempt in range(retries):
        try:
            headers['User-Agent'] = random.choice(user_agents)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            if response.status_code in [403, 404]:
                print(f"Skipping page due to error {response.status_code}: {url}")
                return None
            if 500 <= response.status_code < 600:
                time.sleep(3)  # Wait for 3 seconds before retrying
            else:
                raise
        except Exception as err:
            print(f"An error occurred: {err}")
            time.sleep(3)  # Wait for 3 seconds before retrying
    return None

# Function to save reviews to Excel
def save_to_excel(reviews, page):
    # Store reviews in DataFrame
    df = pd.DataFrame(reviews)

    # Generate filename with timestamp to avoid conflicts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'trustpilot_reviews_up_to_page_{page}_{timestamp}.xlsx'

    # Export DataFrame to Excel file
    try:
        df.to_excel(output_file, index=False)
        print(f"Reviews have been exported to {output_file}")
    except Exception as e:
        print(f"Failed to export reviews to Excel: {e}")

# Loop through each page starting from 401
for page in range(401, max_pages + 1):
    url = f"{base_url}{page}"
    print(f"Fetching page {page}: {url}")
    
    response = fetch_page(url)
    if response is None:
        print(f"Skipping page {page} due to errors.")
        continue

    # Parse HTML content with BeautifulSoup
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        print("Page content fetched successfully.")
    except Exception as e:
        print(f"Failed to parse page {page} content: {e}")
        continue

    # Find review elements (update the class names based on the actual page)
    try:
        reviews = soup.find_all('article', class_='styles_reviewCard__hcAvl')
        print(f"Number of reviews found: {len(reviews)}")
    except Exception as e:
        print(f"Failed to find reviews on page {page}: {e}")
        continue

    if reviews:
        # Iterate through each review and extract content
        for review in reviews:
            try:
                # Get review text
                review_text_element = review.find('p', class_='typography_body-l__KUYFJ')
                review_text = review_text_element.text.strip() if review_text_element else 'No review text'

                # Get rating
                rating = 'No rating'
                rating_element = review.find('div', class_='styles_reviewHeader__iU9Px')
                if rating_element:
                    rating_img = rating_element.find('img')
                    if rating_img and 'alt' in rating_img.attrs:
                        rating_str = rating_img['alt'].split()[1]  # Extract the numeric rating from the alt attribute
                        try:
                            rating = int(rating_str)
                        except ValueError:
                            rating = 'No rating'

                # Get review time and format it
                time_element = review.find('time')
                review_time = time_element['datetime'] if time_element else 'No time'
                if review_time != 'No time':
                    review_time = review_time.split('T')[0]  # Extract the date part

                reviews_list.append({
                    'Review': review_text,
                    'Rating': rating,
                    'Review Date': review_time
                })

                # Print part of the fetched review for debugging
                print(f"Review: {review_text}")
                print(f"Rating: {rating}")
                print(f"Review Date: {review_time}")
            except Exception as e:
                print(f"Failed to process a review on page {page}: {e}")
    else:
        print(f"No reviews found on page {page}")

    # Save to Excel every 100 pages
    if (page - 401 + 1) % 100 == 0:  # interruption happened in page 401, start from page 401
        save_to_excel(reviews_list, page)
        reviews_list.clear()  # Clear the list to start fresh for the next batch

    # Friendly scraping, avoid getting blocked
    time.sleep(3)  # Wait for 3 seconds

# Save any remaining reviews after the loop ends
if reviews_list:
    save_to_excel(reviews_list, page)

print("Finished fetching all pages.")
