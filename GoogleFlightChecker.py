from playwright.sync_api import sync_playwright
import subprocess
import time
import re

# Flight Search Config
ORIGIN = "AUS"  # Austin Airport Code
DESTINATION = "TYO"  # Tokyo (Covers HND/NRT)
MAX_PRICE = 900
DEPARTURE_DATES = ["2025-05-06", "2025-05-07"]  # Check both May 6 & 7
RETURN_DATE = "2025-05-20"

# Your iMessage Number (Must be linked to iMessage on Mac)
YOUR_PHONE_NUMBER = "3468188055"

def send_imessage(phone_number, message):
    """Send an iMessage using AppleScript."""
    script = f'''
    tell application "Messages"
        send "{message}" to buddy "{phone_number}" of (service 1 whose service type is iMessage)
    end tell
    '''
    subprocess.run(["osascript", "-e", script])

def extract_price(text):
    """Extract numerical price from text (e.g., "$850" -> 850)."""
    match = re.search(r"\$\d+", text)
    return int(match.group()[1:]) if match else None

def check_flights():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for departure_date in DEPARTURE_DATES:
            # Generate Correct Google Flights Link
            flights_url = f"https://www.google.com/travel/flights/search?tfs=CBwQAhojEgoyMDI1LTA1-{departure_date[-2:]}agcIARIDVVNqDAgDEgoyMDI1LTA1-20agcIARIDVFlO&tfu=EgYIARAAGAA"
            
            # Open Google Flights
            page.goto(flights_url)
            time.sleep(5)  # Allow time for the page to load
            
            # Scrape prices
            flights = page.locator("div[aria-label*='round trip from Austin']").all()
            
            for flight in flights:
                price_text = flight.locator("div[class*='YMlIz']").text_content()
                price = extract_price(price_text)

                if price and price < MAX_PRICE:
                    print(f"✅ Flight found for ${price} ({departure_date} - {RETURN_DATE}): {flights_url}")
                    
                    # Send iMessage Alert
                    send_imessage(YOUR_PHONE_NUMBER, f"🚀 Flight Found: ${price} round-trip (Depart: {departure_date})! Book now: {flights_url}")
                    
                    browser.close()
                    return True
        
        print("❌ No flights found under $900 for both dates.")
        browser.close()
        return False

if __name__ == "__main__":
    check_flights()
