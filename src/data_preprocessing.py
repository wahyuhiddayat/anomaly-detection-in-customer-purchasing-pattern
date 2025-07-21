import pandas as pd
import re
import warnings
import os

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_missing_values(df):
    """Clean missing values and handle customer data."""
    df_clean = df.copy()
    
    # Remove rows with missing Description
    df_clean = df_clean.dropna(subset=["Description"])
    
    # Create CustomerType and handle missing CustomerID
    df_clean['CustomerType'] = df_clean['CustomerID'].apply(lambda x: 'Guest' if pd.isna(x) else 'Registered')
    df_clean['CustomerID'] = df_clean.apply(lambda row: f"Guest_{row['InvoiceNo']}" if pd.isna(row['CustomerID']) else row['CustomerID'], axis=1)
    
    return df_clean

def handle_duplicates(df):
    """Handle duplicate records by grouping and aggregating."""
    df_clean = df.groupby(['InvoiceNo', 'StockCode', 'UnitPrice'], as_index=False).agg({
        'Quantity': 'sum',
        'InvoiceDate': 'first',
        'CustomerID': 'first',
        'Country': 'first',
        'Description': 'first'
    })
    
    return df_clean

def convert_data_types(df):
    """Convert data types to appropriate formats."""
    df_clean = df.copy()
    
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
    df_clean['UnitPrice'] = df_clean['UnitPrice'].astype(str).str.replace(',', '.').astype(float)
    
    return df_clean

def filter_data(df):
    """Filter out unwanted records."""
    df_clean = df.copy()
    
    # Remove "Unspecified" country
    df_clean = df_clean[df_clean["Country"] != "Unspecified"]
    
    # Remove refund and adjust bad debt transactions
    df_clean = df_clean[~df_clean["InvoiceNo"].str.startswith(("C", "A"))]
    
    # Remove negative quantities
    df_clean = df_clean[df_clean["Quantity"] > 0]
    
    # Remove negative unit prices
    df_clean = df_clean[df_clean["UnitPrice"] > 0]
    
    return df_clean

def normalize_text_fields(df):
    """Normalize text fields by trimming whitespace and standardizing case."""
    df_clean = df.copy()
    
    # Trim whitespace and uppercase for Description
    df_clean['Description'] = df_clean['Description'].str.strip().str.upper()
    
    # Trim whitespace for Country
    df_clean['Country'] = df_clean['Country'].str.strip()
    
    return df_clean

def category_extraction(description):
    """Extract product category from description."""
    if not isinstance(description, str):
        return "unknown"

    # Convert to lowercase and remove extra spaces
    desc = description.lower().strip()

    # Normalize text, remove symbols, size numbers, and double spaces
    desc = re.sub(r"[^\w\s]", " ", desc)       # remove symbols like "/ , . ( )"
    desc = re.sub(r"s/\d+", "", desc)          # remove formats like "S/3"
    desc = re.sub(r"\b\d+('s)?\b", "", desc)   # remove standalone numbers (e.g., "15", "3")
    desc = re.sub(r"\s+", " ", desc)           # replace double spaces with single space

    # Strong pattern detection (high priority), return immediately
    strong_patterns = {
        r"\bchopsticks\b": "chopsticks",
        r"\bmilk pan\b": "milk pan",
        r"\bpencils?\b": "pencils",
        r"\bpen\b": "pen",
        r"\bwrap\b": "gift wrap",
        r"\btv dinner tray\b": "tray",
        r"\b(breakfast set|lunch set|dinner set)\b": "dining set",
        r"\bdrawer knob\b": "drawer knob",
        r"\bnecklace\b": "necklace",
        r"\bearrings?\b": "earrings",
        r"\bbracelet\b": "bracelet",
        r"\bt[ -]lights?\b": "t-lights",
        r"\bsoft toy\b": "soft toy",
        r"\bcake tins\b": "cake tins",
        r"\bherb tins\b": "herb tins",
        r"\bspice tins\b": "spice tins",
        r"\bbaking mould\b": "baking mould",
        r"\bribbons\b": "ribbons",
        r"\bphoto frame\b": "photo frame",
        r"\bdanish rose\b": "danish rose collection",
        r"\bsewing box\b": "sewing box"
    }

    for pattern, category in strong_patterns.items():
        if re.search(pattern, desc):
            return category

    # Detect special collections
    collections = {
        "danish rose": "danish rose collection"
    }

    for collection, category in collections.items():
        if collection in desc:
            return category

    # Handle descriptions containing "set"
    if "set of" in desc or "set " in desc:
        # T-light
        if "t light" in desc or "t-light" in desc:
            return "t-lights"
        # Cake/herb/spice tins
        if "cake tin" in desc:
            return "cake tins"
        if "herb tin" in desc:
            return "herb tins"
        if "spice tin" in desc:
            return "spice tins"
        # Flying ducks
        if "flying duck" in desc:
            return "flying ducks"
        # Gift wrap
        if "gift wrap" in desc:
            return "gift wrap"
        # Cutlery
        if "cutlery" in desc:
            return "cutlery"

    # Detect "metal sign" items
    if re.search(r"\bmetal sign\b", desc):
        return "sign"

    # Priority product phrases
    keywords = [
        "money bank", "t light holder", "tea towel", "mug cosy", "bird ornament",
        "playhouse bedroom", "playhouse kitchen", "hot water bottle", "napkin charms",
        "building block word", "inflatable globe", "shopping bag", "hand warmer",
        "cake stand", "heart t light", "bathroom set", "lunch box", "sewing kit",
        "writing set", "glass cloche", "jigsaw puzzle", "mini jigsaw", "cabinet",
        "parasol", "mug", "bag", "balloons", "candle", "candlestick", "card holder",
        "heart wicker", "picnic basket", "flying ducks", "decorative plate", "photo frame",
        "trinket trays", "folding chair", "soft toy"
    ]
    for phrase in keywords:
        if phrase in desc:
            return phrase

    # Handle patterns like "set of 3 heart tins" -> extract "heart tins"
    match = re.match(r"(box|pack|set) of \d+ ([a-z\s]+)", desc)
    if match:
        product = match.group(2).strip()
        # Some exceptions
        if product in ["heart", "cake", "strawberry"]:
            if "chopstick" in desc:
                return "chopsticks"
        if "ribbon" in product:
            return "ribbons"
        if "t light" in product or "t-light" in product:
            return "t-lights"
        return product

    # Common product words prioritized over themes
    product_words = [
        "pencil", "pen", "chopstick", "drawer", "knob", "pan", "milk", "breakfast",
        "mug", "plate", "bowl", "spoon", "fork", "knife", "towel", "napkin",
        "earring", "frame", "tin", "toy", "baking", "ribbon"
    ]

    for word in product_words:
        if word in desc:
            # Special cases for two-word combinations
            if word == "drawer" and "knob" in desc:
                return "drawer knob"
            if word == "milk" and "pan" in desc:
                return "milk pan"
            if word == "breakfast" and "set" in desc:
                return "breakfast set"
            if word == "earring":
                return "earrings"
            if word == "frame" and "photo" in desc:
                return "photo frame"
            if word == "tin" and "cake" in desc:
                return "cake tins"
            if word == "tin" and "herb" in desc:
                return "herb tins"
            if word == "tin" and "spice" in desc:
                return "spice tins"
            if word == "toy" and "soft" in desc:
                return "soft toy"
            if word == "baking" and "mould" in desc:
                return "baking mould"
            if word == "ribbon":
                return "ribbons"
            if word == "pen":
                return "pen"
            return word

    # Handle themed products (e.g. "christmas candle")
    theme_match = re.search(r'\b(christmas|vintage|retro|classic|modern|valentine|easter)\b', desc)
    if theme_match:
        theme = theme_match.group(1)
        tokens = desc.split()
        for t in tokens:
            if t not in {'the', 'of', 'for', 'and', 'in', 'on', 'with', theme} and len(t) > 2:
                # If product is ribbons with theme -> still use "ribbons"
                if t == "ribbons" and theme == "christmas":
                    return "ribbons"
                return f"{t} {theme}"
        return f"{theme} item"  # fallback if no main product found

    # Handle location-based descriptions
    location_match = re.search(r"(london|paris|england|york|japan)", desc)
    if location_match:
        return f"souvenir {location_match.group(1)}"

    # Detect common ending words used as products
    endings = ['mug', 'parasol', 'box', 'globe', 'sign', 'towel', 'bracelet', 'necklace', 'plate']
    for word in endings:
        if desc.endswith(word):
            return word

    # Common two-word combinations
    word_pairs = [
        "childs breakfast", "picnic basket", "heart wicker", "ceramic drawer",
        "danish rose", "flying ducks", "soft toy"
    ]
    for pair in word_pairs:
        if pair in desc:
            if pair == "ceramic drawer" and "knob" in desc:
                return "drawer knob"
            if pair == "danish rose":
                return "danish rose collection"
            return pair

    # Extract 1-2 main words (excluding colors, sizes, or unimportant words)
    non_substantive = {'the', 'of', 'and', 'in', 'on', 'for', 'with', 'a', 'to', 'set', 'pack', 'mrs', 'mr'}
    color_words = {'red', 'blue', 'green', 'white', 'black', 'pink', 'purple', 'yellow', 'turq', 'turquoise', 'silver'}
    size_words = {'small', 'large', 'mini', 'tall', 'short', 'big', 'medium'}

    words = desc.split()
    content_words = []

    for word in words:
        if (word not in non_substantive
            and word not in color_words
            and word not in size_words
            and len(word) > 2):
            content_words.append(word)

    # Return combination of 1-2 substantive words as fallback
    if len(content_words) >= 2:
        return ' '.join(content_words[:2])
    elif content_words:
        return content_words[0]
    else:
        # Last fallback, take first word that's not color/size
        for word in words:
            if word not in color_words and word not in size_words and len(word) > 2:
                return word
        return words[0] if words else "misc"

def extract_categories(df):
    """Extract product categories from descriptions."""
    df_clean = df.copy()
    df_clean["CategoryExtraction"] = df_clean["Description"].apply(category_extraction)
    return df_clean

def calculate_total_price(df):
    """Calculate total price for each transaction."""
    df_clean = df.copy()
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    return df_clean

def extract_time_features(df):
    """Extract time-based features from InvoiceDate."""
    df_clean = df.copy()
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    return df_clean

def calculate_customer_features(df):
    """Calculate customer-based features."""
    df_clean = df.copy()
    
    # Frequency of purchase per customer
    df_clean["Frequency"] = df_clean.groupby("CustomerID")["InvoiceNo"].transform("nunique")
    
    # Days since last purchase (recency)
    latest_date = df_clean["InvoiceDate"].max()
    df_clean["Recency"] = (latest_date - df_clean["InvoiceDate"]).dt.days
    
    return df_clean

def preprocess_data(url):
    """Main preprocessing pipeline."""
    # Load data
    df = load_data(url)
    
    # Clean missing values
    df = clean_missing_values(df)
    
    # Handle duplicates
    df = handle_duplicates(df)
    
    # Convert data types
    df = convert_data_types(df)
    
    # Filter data
    df = filter_data(df)
    
    # Normalize text fields
    df = normalize_text_fields(df)
    
    # Extract categories
    df = extract_categories(df)
    
    # Calculate total price
    df = calculate_total_price(df)
    
    # Extract time features
    df = extract_time_features(df)
    
    # Calculate customer features
    df = calculate_customer_features(df)
    
    return df

def main():
    """Main execution function."""
    file_path = "data/raw/Online Retail Dataset.xlsx - Online Retail.csv"
    
    # Run preprocessing pipeline
    df_processed = preprocess_data(file_path)
    
    # Create output directory if it doesn't exist
    output_file = "data/processed/cleaned_retail_data.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV file
    df_processed.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    return df_processed

if __name__ == "__main__":
    processed_data = main()