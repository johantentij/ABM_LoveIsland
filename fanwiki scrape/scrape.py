import requests
import pandas as pd

# 1. Define the API endpoint and the specific page you want
# Notice we use api.php instead of /wiki/
wiki_url = "https://loveisland.fandom.com/api.php"

N_seasons = 13
for i in range(1, N_seasons + 1):
    print(f"getting season {i}...")

    # We pass the parameters as a dictionary to make it clean
    params = {
        "action": "parse",
        "page": f"Love_Island_(Season_{i})", # The exact title of the page
        "format": "json"
    }

    # 2. Make the request (Cloudflare usually ignores API endpoints)
    response = requests.get(wiki_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        
        # 3. Extract the raw HTML string from the JSON payload
        try:
            raw_html = data['parse']['text']['*']
            
            # 4. Feed the raw HTML string directly into Pandas!
            # We use the same matching logic we used earlier
            tables = pd.read_html(
                raw_html, 
                attrs={"class": "wikitable"}, 
            )
            
            # Grab the target table
            season_table = tables[0]
            season_table.drop(["Unnamed: 0", "Notes"])
            season_table.to_csv(f"season_{i}_data.csv", index=False, encoding="utf-8-sig")
            
        except KeyError:
            print("Error: Could not find the page content in the JSON. Check the page title.")
        except ValueError:
            print("Error: Pandas could not find a table matching your criteria in the HTML.")
    else:
        print(f"API request failed with status code: {response.status_code}")