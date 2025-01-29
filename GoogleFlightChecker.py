from playwright.sync_api import sync_playwright
import subprocess
import time
import re

# ✅ Corrected Google Flights URLs
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

def extract_flight_details(page):
    """Scrape only the departure time, airline names, and prices from Google Flights."""
    flights = []
    
    # Wait for elements to load
    page.wait_for_timeout(5000)

    # Locate elements for departure time, airline, and price
    price_elements = page.locator("div[class*='YMlIz']").all()
    airline_elements = page.locator("div[class*='sSHqwe']").all()  # Precise airline selector
    time_elements = page.locator("div[class*='Ir0Voe']").all()  # Departure time selector

    print(f"🔍 Found {len(price_elements)} price elements, {len(airline_elements)} airline elements, and {len(time_elements)} time elements")

    for price_elem, airline_elem, time_elem in zip(price_elements, airline_elements, time_elements):
        price_text = price_elem.text_content().strip()
        airline_text = airline_elem.text_content().strip()
        time_text = time_elem.text_content().strip()

        # Extract price using regex and ensure it's a valid price
        price_match = re.search(r"\$(\d+)", price_text)
        if price_match:
            price = int(price_match.group(1))

            # Ignore non-price text (e.g., "1 stop") by checking for "$"
            if "$" in price_text:
                # **Format the extracted time to remove duplicate/unwanted text**
                time_clean = re.match(r"(\d{1,2}:\d{2} [AP]M)", time_text)
                if time_clean:
                    time_final = time_clean.group(1)  # Get only the "6:00 AM" part

                    # **Format the extracted airline name (remove junk text)**
                    airline_final = airline_text.split(" - ")[0]  # Remove extra info if any

                    flights.append((time_final, airline_final, price))

    return flights

def check_flights():
    """Scrape Google Flights and list only departure times, airlines, and prices under $900."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Change to False to see browser actions
        page = browser.new_page()

        all_cheapest_flights = []  # Store all flights under $900

        for departure_date, flights_url in FLIGHT_URLS.items():
            print(f"🔍 Checking flights for {departure_date} - 2025-05-20")
            page.goto(flights_url)
            page.wait_for_load_state("networkidle")  # Wait for full page load
            
            # Extract all airline names and prices
            flights = extract_flight_details(page)

            if not flights:
                print(f"⚠️ No flights detected for {departure_date}, check the selectors!")

            # Print extracted flights for debugging
            print(f"📋 Extracted Flights on {departure_date}: {flights}")

            # Filter flights under $900
            cheap_flights = [(time, airline, price) for time, airline, price in flights if price < MAX_PRICE]

            if cheap_flights:
                print(f"✅ Found flights under $900 on {departure_date}: {cheap_flights}")

                formatted_flights = "\n".join([f"{time} - {airline} - ${price}" for time, airline, price in cheap_flights])
                all_cheapest_flights.append(f"📅 {departure_date}\n{formatted_flights}")

        browser.close()

        if all_cheapest_flights:
            # Format message for iMessage
            message = "🚀 Flights Under $900 Found!\n\n" + "\n\n".join(all_cheapest_flights)
            print(f"📩 Sending iMessage:\n{message}")
            send_imessage(YOUR_PHONE_NUMBER, message)
        else:
            print("❌ No flights found under $900 for both dates.")

if __name__ == "__main__":
    check_flights()
