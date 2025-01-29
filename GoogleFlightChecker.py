from playwright.sync_api import sync_playwright
import subprocess
import time
import re

# Google Flights URLs for May 6-20 and May 7-20
FLIGHT_URLS = {
    "2025-05-06": "https://www.google.com/travel/flights/search?tfs=CBwQAhonEgoyMDI1LTA1LTA2agsIAhIHL20vMHZ6bXIMCAISCC9tLzA3ZGZrGicSCjIwMjUtMDUtMjBqDAgCEggvbS8wN2Rma3ILCAISBy9tLzB2em1AAUgBcAGCAQsI____________AZgBAQ&tfu=EgoIABAAGAAgAigB",
    "2025-05-07": "https://www.google.com/travel/flights/search?tfs=CBwQAhonEgoyMDI1LTA1LTA3agsIAhIHL20vMHZ6bXIMCAISCC9tLzA3ZGZrGicSCjIwMjUtMDUtMjBqDAgCEggvbS8wN2Rma3ILCAISBy9tLzB2em1AAUgBcAGCAQsI____________AZgBAQ&tfu=EgYIACACKAEiAxIBMA"
    }
MAX_PRICE = 900  # Set the max price for alerts
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
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for departure_date, flights_url in FLIGHT_URLS.items():
            print(f"🔍 Checking flights for {departure_date} - 2025-05-20")
            page.goto(flights_url)
            time.sleep(5)  # Allow time for the page to load
            
            # Extract all flight prices
            prices = extract_prices(page)
            
            # Check if any flight is under $900
            for price in prices:
                if price < MAX_PRICE:
                    print(f"✅ Flight found for ${price} ({departure_date} - 2025-05-20): {flights_url}")
                    
                    # Send iMessage Alert
                    send_imessage(YOUR_PHONE_NUMBER, f"🚀 Flight Found: ${price} round-trip (Depart: {departure_date})! Book now: {flights_url}")
                    
                    browser.close()
                    return True
        
        print("❌ No flights found under $900 for both dates.")
        browser.close()
        return False

if __name__ == "__main__":
    check_flights()
