from playwright.sync_api import sync_playwright
import subprocess
import time
import re

# Google Flights URLs for May 6-20 and May 7-20
FLIGHT_URLS = {
    "2025-05-06": "https://www.google.com/travel/flights/search?tfs=CBwQAhonEgoyMDI1LTA1LTA2agsIAhIHL20vMHZ6bXIMCAMSCC9tLzA3ZGZrGicSCjIwMjUtMDUtMjBqDAgDEggvbS8wN2Rma3ILCAISBy9tLzB2em1AAUgBcAGCAQsI____________AZgBAQ&tfu=EgoIABAAGAAgAigB&hl=en-US&gl=US",
    "2025-05-07": "https://www.google.com/travel/flights/search?tfs=CBwQAhonEgoyMDI1LTA1LTA3agsIAhIHL20vMHZ6bXIMCAMSCC9tLzA3ZGZrGicSCjIwMjUtMDUtMjBqDAgDEggvbS8wN2Rma3ILCAISBy9tLzB2em1AAUgBcAGCAQsI____________AZgBAQ&tfu=EgoIABAAGAAgAigB&hl=en-US&gl=US"
}
MAX_PRICE = 900  # Max price threshold

# Your iMessage Number (Must be linked to iMessage on Mac)
YOUR_PHONE_NUMBER = "3468188055"  # Replace with your phone number

def send_imessage(phone_number, message):
    """Send an iMessage using AppleScript."""
    script = f'''
    tell application "Messages"
        send "{message}" to buddy "{phone_number}" of (service 1 whose service type is iMessage)
    end tell
    '''
    subprocess.run(["osascript", "-e", script])

def extract_prices(page):
    """Scrape all flight prices from Google Flights page."""
    prices = []
    price_elements = page.locator("div[class*='YMlIz']").all()

    for element in price_elements:
        price_text = element.text_content()
        match = re.search(r"\$(\d+)", price_text)  # Extract numbers after $
        if match:
            prices.append(int(match.group(1)))

    return prices

def check_flights():
    """Scrape Google Flights and list all flights under $900."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        all_cheapest_flights = []  # Store all flights under $900

        for departure_date, flights_url in FLIGHT_URLS.items():
            print(f"🔍 Checking flights for {departure_date} - 2025-05-20")
            page.goto(flights_url)
            time.sleep(5)  # Allow time for the page to load
            
            # Extract all flight prices
            prices = extract_prices(page)
            
            # Find flights under $900
            cheap_flights = [price for price in prices if price < MAX_PRICE]

            if cheap_flights:
                print(f"✅ Found flights under $900 on {departure_date}: {cheap_flights}")
                all_cheapest_flights.append(f"{departure_date}: {', '.join([f'${p}' for p in cheap_flights])} - {flights_url}")

        browser.close()

        if all_cheapest_flights:
            # Format message for iMessage
            message = "🚀 Flights Under $900 Found!\n\n" + "\n".join(all_cheapest_flights)
            send_imessage(YOUR_PHONE_NUMBER, message)
        else:
            print("❌ No flights found under $900 for both dates.")

if __name__ == "__main__":
    check_flights()
