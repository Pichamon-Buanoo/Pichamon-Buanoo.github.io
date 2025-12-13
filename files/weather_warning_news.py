import requests
import pandas as pd
import xml.etree.ElementTree as ET
import textwrap # Used for formatting long text

# --- 1. Define API Key and URL ---
# Use the same key you used in the original code (placeholder if not using a real key)
API_UID = "api" 
API_UKEY = "api12345"
# API No. 10: Weather Warning News
URL = f"https://data.tmd.go.th/api/WeatherWarningNews/v1/?uid={API_UID}&ukey={API_UKEY}"

def get_xml_text(element, tag, default_val="N/A"):
    """Safely retrieves the text content of a sub-element, stripping whitespace."""
    if element is None:
        return default_val
        
    node = element.find(tag)
    # Strip whitespace to ensure clean text retrieval
    return node.text.strip() if node is not None and node.text else default_val

def get_warning_news():
    """
    Fetches weather warning news from TMD API (No. 10) and extracts key details.
    Returns a pandas DataFrame of the warnings.
    """
    records = []
    
    try:
        print(f"--- Fetching Weather Warning News (API No. 10) from {URL} ---")
        response = requests.get(URL, timeout=15)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status() 
        root = ET.fromstring(response.content)
        
        # Check if any warning data exists
        warnings = root.findall('Warning')
        if not warnings:
            print("No active warning news found.")
            return pd.DataFrame()

        for warning in warnings:
            records.append({
                'Issue_No': get_xml_text(warning, 'IssueNo'),
                'Announce_Date': get_xml_text(warning, 'AnnounceDate'),
                'Effect_Start': get_xml_text(warning, 'EffectStartDate'),
                'Effect_End': get_xml_text(warning, 'EffectEndDate'),
                'Title_Thai': get_xml_text(warning, 'TitleThai'),
                'Headline_Thai': get_xml_text(warning, 'HeadlineThai'),
                'Description_Thai': get_xml_text(warning, 'DescriptionThai'),
                'Web_URL_Thai': get_xml_text(warning, 'WebUrlThai'),
            })
        
        # Convert to DataFrame and sort by the latest announcement date
        df = pd.DataFrame(records)
        df['Announce_Date'] = pd.to_datetime(df['Announce_Date'], errors='coerce')
        df = df.sort_values(by='Announce_Date', ascending=False).reset_index(drop=True)
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Network or API Error: {e}. Check URL/Key/Internet connection.")
        return pd.DataFrame()
    except ET.ParseError:
        print("XML Parsing Error: The response is not a valid XML format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def display_warning_news(df):
    """
    Displays the fetched news data, showing a summary table and the full text of the latest warning.
    """
    if df.empty:
        return

    print("\n" + "="*80)
    print("📢 Summary of All Weather Warning News (Sorted by Latest)")
    print("="*80)
    
    # --- 1. Display Summary Table (Analogous to Plotting) ---
    summary_df = df[['Issue_No', 'Announce_Date', 'Title_Thai', 'Headline_Thai', 'Effect_Start', 'Effect_End']].copy()
    
    # Trim columns for better console display
    summary_df['Title_Thai'] = summary_df['Title_Thai'].str.slice(0, 50) + '...'
    summary_df['Headline_Thai'] = summary_df['Headline_Thai'].str.slice(0, 70) + '...'
    
    print("\n--- Warning News Summary Table ---\n")
    print(summary_df.to_string(index=False))
    
    # --- 2. Display Latest News Details (Analogous to Colorbar/Values) ---
    latest_news = df.iloc[0]
    
    print("\n" + "="*80)
    print(f"🚨 Details of the Latest Issue (Issue No: {latest_news['Issue_No']})")
    print("="*80)
    
    # Function to wrap long text for console readability
    def wrap_text(text, width=78):
        return "\n".join(textwrap.wrap(text, width=width))

    print(f"Announcement Date: {latest_news['Announce_Date']}")
    print(f"Effect Period: {latest_news['Effect_Start']} to {latest_news['Effect_End']}")
    print("-" * 50)
    print("Warning Title:")
    print(wrap_text(latest_news['Title_Thai']))
    print("\nHeadline:")
    print(wrap_text(latest_news['Headline_Thai']))
    print("\nFull Description:")
    print(wrap_text(latest_news['Description_Thai']))
    print("-" * 50)
    print(f"Full Document URL: {latest_news['Web_URL_Thai']}")
    print("\n")


if __name__ == "__main__":
    df_warnings = get_warning_news()
    if not df_warnings.empty:
        display_warning_news(df_warnings)
    else:
        print("Finished process, but no data was displayed.")