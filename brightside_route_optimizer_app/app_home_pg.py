import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import PyPDF2
import re
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import logging
import requests
import concurrent.futures
import boto3
from io import StringIO

# Configure basic logging for warnings and errors only
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize all session state variables
def initialize_session_state():
    """Initialize all session state variables used in the application."""
    # Tracks the current step in the route generation process (1-5)
    # Step 1: Select Team Members
    # Step 2: Upload Route Document
    # Step 3: Review Routes
    # Step 4: Assign Routes
    # Step 5: Confirmation and Email Templates
    if 'step' not in st.session_state:
        st.session_state.step = 1

    # Stores the list of email addresses for selected team members
    # Used to track which team members will be assigned routes
    if 'selected_emails' not in st.session_state:
        st.session_state.selected_emails = []

    # Stores the list of addresses extracted from the uploaded PDF
    # These are the delivery locations that need to be routed
    if 'addresses' not in st.session_state:
        st.session_state.addresses = []

    # Stores the optimized route groups after clustering and optimization
    # Each group represents a set of addresses that will be assigned to one team member
    if 'route_groups' not in st.session_state:
        st.session_state.route_groups = []

    # Maps team member emails to their assigned route groups
    # Used to track which routes are assigned to which team members
    if 'assignments' not in st.session_state:
        st.session_state.assignments = {}

    # Stores the uploaded PDF file containing delivery addresses
    # Used for address extraction and processing
    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None

    # Flag indicating whether route optimization is currently in progress
    # Used to disable UI elements during optimization
    if 'is_optimizing' not in st.session_state:
        st.session_state.is_optimizing = False

    # Flag indicating whether to show the email confirmation dialog
    # Used to control the display of the confirmation key input
    if 'show_confirmation' not in st.session_state:
        st.session_state.show_confirmation = False

    # Cache for storing traffic information from Google Maps API
    # Key: tuple of addresses, Value: traffic data for that route
    if 'traffic_info_cache' not in st.session_state:
        st.session_state.traffic_info_cache = {}

    # Cache for storing processed traffic information for each route group
    # Key: route group index, Value: traffic data for that route
    if 'cached_traffic_info' not in st.session_state:
        st.session_state.cached_traffic_info = {}

    # Cache for storing geocoded coordinates for addresses
    # Key: address, Value: (latitude, longitude) tuple
    if 'cached_coordinates' not in st.session_state:
        st.session_state.cached_coordinates = {}

    # Stores statistics for each route group
    # Includes number of addresses, total time, and other metrics
    if 'route_stats' not in st.session_state:
        st.session_state.route_stats = []

    # Records address movements during route optimization
    # Used for logging and debugging route balancing
    if 'address_movements' not in st.session_state:
        st.session_state.address_movements = []

    # Stores the index of a team member to be deleted
    # Used in the team member management interface
    if 'member_to_delete_index' not in st.session_state:
        st.session_state.member_to_delete_index = None

# AWS Configuration
# Note: These secrets should ideally be accessed in the main app_main.py
# and passed down or accessed carefully in pages. Keeping them here for now.
aws_access_key_id = st.secrets.get("aws_access_key_id")
aws_secret_access_key = st.secrets.get("aws_secret_access_key")
aws_region = st.secrets.get("aws_region")

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# S3 bucket and file configuration
TEAM_MEMBERS_BUCKET = "brightside-route-optimizer"
TEAM_MEMBERS_FILE = "brightside_team_members_dummy_file.csv" # Using dummy file


def load_team_members():
    """Load team members from S3 CSV file."""
    try:
        # Get the CSV file from S3
        response = s3.get_object(Bucket=TEAM_MEMBERS_BUCKET, Key=TEAM_MEMBERS_FILE)
        csv_content = response['Body'].read().decode('utf-8')

        # Read CSV into DataFrame
        df = pd.read_csv(StringIO(csv_content))

        # Convert to dictionary
        team_members_dict = dict(zip(df['name'], df['email']))
        return team_members_dict
    except Exception as e:
        logger.error(f"Error loading team members from S3 file s3://{TEAM_MEMBERS_BUCKET}/{TEAM_MEMBERS_FILE}: {str(e)}")
        # Fallback to empty dict if S3 load fails
        return {}

# Load team members from S3
EMAILS = load_team_members()

# Email configuration
GMAIL_USER = st.secrets.get("GMAIL_USER")
GMAIL_APP_PASSWORD = st.secrets.get("GMAIL_APP_PASSWORD")

# --- Google Maps API Key Setup ---
# To use drive-time optimization, set your Google Maps Directions API key in your .streamlit/secrets.toml file:
# [google]
# directions_key = "YOUR_API_KEY_HERE"
# The variable below will load it automatically.
GGL_DIRECTIONS_KEY = st.secrets.get("GGL_DIRECTIONS_KEY")
BRIGHTSIDE_CONFIRMATION_KEY = st.secrets.get("BRIGHTSIDE_CONFIRMATION_KEY")


# Helper functions
def extract_addresses_from_pdf(pdf_file):
    """Extract address-like text from a PDF, approximating the Address column"""
    addresses = set()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        temp_path = temp_file.name

    try:
        # Open PDF file in binary read mode
        with open(temp_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Loop through each page in the PDF
            for page in pdf_reader.pages:
                # Extract all text from current page
                text = page.extract_text()

                # Split text into individual lines
                lines = text.split('\n')

                # Process each line of text
                for line in lines:
                    # Remove leading/trailing whitespace
                    line = line.strip()

                    # Skip empty lines or lines with unwanted keywords
                    if not line or "Address" in line or "Delivery" in line:
                        continue

                    # Define regex pattern to match complete addresses
                    # Pattern looks for: [Building Name - ] Street Number Street Name, City, State ZIP
                    pattern = (
                        r'(?:([A-Za-z\s]+(?:Towers?|Building|Center|Plaza)) - )?'  # Optional building name with dash
                        r'(\d{3,5})\s+'                                           # Street number (3-5 digits) + space  
                        r'([\w\s#/.,-]+?(?:Ave|St|Blvd|Rd|Dr|Ct|Ln|Way|Place|Pl|Trail|Parkway|Pkwy|Cir|Terrace)[\w\s]*?)'  # Street name + type
                        r',\s*([A-Za-z\s]+)'                                      # Comma + city name
                        r',\s*([A-Z]{2})'                                         # Comma + state (2 letters)
                        r'\s+(\d{5}(?:-\d{4})?)'                                  # Space + ZIP code (5 or 9 digits)
                    )

                    # Search for the address pattern in current line (case insensitive)
                    match = re.search(pattern, line, re.IGNORECASE)

                    # If pattern is found, extract the address components
                    if match:
                        # building = match.group(1)    # Building name (could be None)
                        number = match.group(2)      # Street number
                        street = match.group(3)      # Street name and type
                        city = match.group(4)        # City name
                        state = match.group(5)       # State abbreviation
                        zip_code = match.group(6)    # ZIP code

                        # Build Google Maps compatible address (always exclude building name)
                        # Format: "Street Number Street Name, City, State ZIP"
                        google_maps_address = f"{number} {street.strip()}, {city.strip()}, {state} {zip_code}"

                        # Clean up any double spaces
                        google_maps_address = re.sub(r'\s+', ' ', google_maps_address)

                        # Add address to set (automatically removes duplicates)
                        addresses.add(google_maps_address)

        # Print results
        print(f"Found {len(addresses)} unique addresses:")
        for address in sorted(addresses):
            print(address)

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")
    finally:
        os.unlink(temp_path)

    return sorted(addresses)

# Create a single geolocator instance to reuse
geolocator = Nominatim(user_agent="route_optimizer_v1.0", timeout=10)  # Use Nominatim geocoder with increased timeout and rate limiting


# get coordinates function
def get_coordinates(address):
    """Get coordinates (latitude, longitude) for an address using geopy."""
    try:
        # Input validation
        if not address or not isinstance(address, str):
            return None

        address = address.strip()
        if not address:
            return None

        # Check session state cache first
        if 'cached_coordinates' not in st.session_state:
            st.session_state.cached_coordinates = {}

        if address in st.session_state.cached_coordinates:
            return st.session_state.cached_coordinates[address]

        # Add Minnesota to the address if not already present
        if "Minnesota" not in address and "MN" not in address:
            address = f"{address}, Minnesota, USA"

        # Rate limiting
        time.sleep(1.5)

        location = geolocator.geocode(address)

        if location:
            coords = (location.latitude, location.longitude)
            st.session_state.cached_coordinates[address] = coords
            return coords
        else:
            st.warning(f"Could not geocode address: {address}")
            st.session_state.cached_coordinates[address] = None
            return None

    except Exception as e:
        st.warning(f"Error geocoding address '{address}': {e}")
        time.sleep(2)
        return None


def optimize_routes(addresses, num_groups):
    """
    Divide addresses into optimized route groups using a combination of KMeans clustering and drive-time optimization.

    Special Rules:
    1. If a group contains '1920 4th Ave S', limit that group to 3 addresses (including it). This is because the driver who gets this route has to sit at this location to distribute bags
    2. Any extra addresses in that group are redistributed to the nearest other group

    Algorithm Steps:
    1. Geocode all addresses to get coordinates
    2. Use KMeans clustering to create initial groups based on geographic proximity
    4. Optimize each group's route order using Google Directions API
    5. Balance routes by redistributing addresses if route times are significantly unbalanced

    Optimization Process:
    The algorithm uses a two-phase approach to optimize routes:

    Phase 1 - Geographic Clustering (K-means):
    - Uses latitude/longitude coordinates to create initial groups
    - Groups addresses that are geographically close to each other
    - This is a quick first pass to get reasonable starting groups
    - Special case: If '1920 4th Ave S' is in a group, that group is limited to 3 addresses

    Phase 2 - Drive-Time Optimization (Google Maps API):
    - Takes the groups from Phase 1
    - For each group:
      - Uses Google Directions API to find the optimal driving order
      - Calculates actual drive times between addresses
      - Optimizes the route to minimize total drive time
    - Then attempts to balance routes by:
      - Calculating average drive time across all routes
      - If any route is >30% longer than average:
        - Moves addresses from the longest route to the shortest route
        - Re-optimizes both affected routes using Google Directions API

    The two-phase approach is efficient because:
    1. K-means clustering is fast and gives us a good starting point
    2. Google Directions API is more accurate but slower, so we use it for final optimization
    3. The special case for '1920 4th Ave S' is handled in Phase 1, ensuring that route stays manageable for the driver who needs to distribute bags there

    Route Balancing Details:
    The algorithm uses an iterative approach to balance routes:
    1. Initial Balance Check:
       - Calculates average drive time across all routes
       - Identifies routes that are >30% longer than average
       - Routes are considered balanced when all are within 30% of average

    2. Multi-Attempt Balancing:
       - Makes up to 3 attempts to balance routes
       - Each attempt can move multiple addresses at once
       - Stops early if routes become balanced

    3. Address Movement Rules:
       - Can move up to 1/4 of addresses from a long route
       - Never reduces a route below 3 addresses
       - Moves addresses to the shortest route
       - Re-optimizes both affected routes after each move

    4. Size Protection:
       - Minimum route size: 3 addresses
       - Maximum addresses moved per attempt: min(route_size - 3, route_size // 4)
       - Example: For a route with 8 addresses:
         * route_size - 3 = 5 (can move up to 5 while keeping 3)
         * route_size // 4 = 2 (one quarter of 8)
         * min(5, 2) = 2 (we take the smaller number)
         * Therefore, can move up to 2 addresses
         * Will always keep at least 3 addresses
         * Will never move more than 1/4 of the addresses

    5. Balance Verification:
       - Tracks all addresses throughout the process
       - Verifies no addresses are lost during optimization
       - Warns if any addresses are missing from final routes
    """
    # Validate Google API key
    if not GGL_DIRECTIONS_KEY:
        st.error("Google Maps API key is not configured. Set GGL_DIRECTIONS_KEY in your .streamlit/secrets.toml file.")
        return []

    # Get coordinates for all addresses using parallel processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("Getting coordinates for addresses... (this may take a moment)")

    coordinates = []
    valid_addresses = []

    # Use ThreadPoolExecutor for parallel geocoding
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all geocoding tasks
        future_to_address = {
            executor.submit(get_coordinates, address): address
            for address in addresses
        }

        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_address):
            address = future_to_address[future]
            try:
                coords = future.result()
                if coords:
                    coordinates.append(coords)
                    valid_addresses.append(address)
            except Exception as e:
                st.warning(f"Error geocoding address {address}: {e}")

            completed += 1
            progress_bar.progress(completed / len(addresses))

    progress_bar.empty()

    if not coordinates:
        status_text.error("Could not geocode any addresses. Please check the address format and try again.")
        return []

    status_text.info(f"Successfully geocoded {len(coordinates)} out of {len(addresses)} addresses")

    if len(coordinates) < num_groups:
        status_text.warning(f"Not enough valid addresses ({len(coordinates)}) for the requested number of groups ({num_groups}). Creating {len(coordinates)} groups instead.")
        num_groups = max(1, len(coordinates))

    # Step 2: Apply K-means clustering for initial grouping
    status_text.info("Creating initial route groups...")

    # Coordinates for starting location
    start_coord = get_coordinates("693 Raymond Ave, St Paul, MN 55114")

    # Create feature set: lat, lon, and distance from start
    features = []
    for coord in coordinates:
        dist = geodesic(start_coord, coord).miles
        features.append([coord[0], coord[1], dist])  # lat, lon, distance from start

    X = np.array(features)

    # Run KMeans on the enriched feature set
    kmeans = KMeans(
        n_clusters=num_groups,
        random_state=42,
        n_init=5,
        max_iter=100,
        algorithm='elkan'
    )
    labels = kmeans.fit_predict(X)

    # Group addresses by their cluster assignments
    groups = [[] for _ in range(num_groups)]
    group_coords = [[] for _ in range(num_groups)]
    for i, label in enumerate(labels):
        if i < len(valid_addresses):
            groups[label].append(valid_addresses[i])
            group_coords[label].append(coordinates[i])

    # Remove any empty groups that might have been created
    groups = [group for group in groups if group]
    group_coords = [coords for coords in group_coords if coords]

    # Step 3: Special handling for '1920 4th Ave S'
    SPECIAL_ADDR = '1920 4th Ave S'
    special_pattern = re.compile(r'1920\s*4th\s*Avenue?\s*S', re.IGNORECASE)
    extras = []
    extras_coords = []

    # Find which group contains the special address
    special_group_idx = None
    for idx, group in enumerate(groups):
        if any(special_pattern.search(addr.replace('.', '').replace(',', '')) for addr in group):
            special_group_idx = idx
            break

    if special_group_idx is not None:
        group = groups[special_group_idx]
        group_coords_list = group_coords[special_group_idx]

        # Find the special address and its coordinates
        special_idx = None
        for i, addr in enumerate(group):
            if special_pattern.search(addr.replace('.', '').replace(',', '')):
                special_idx = i
                break

        if special_idx is not None and len(group) > 3:
            # Keep the special address and find 2 closest others
            special_coord = group_coords_list[special_idx]
            distances = []

            for i, coord in enumerate(group_coords_list):
                if i != special_idx:
                    dist = geodesic(special_coord, coord).miles
                    distances.append((i, dist))

            distances.sort(key=lambda x: x[1])
            keep_indices = {special_idx} | {i for i, _ in distances[:2]}

            for i in range(len(group)):
                if i not in keep_indices:
                    extras.append(group[i])
                    extras_coords.append(group_coords_list[i])

            groups[special_group_idx] = [group[i] for i in keep_indices]
            group_coords[special_group_idx] = [group_coords_list[i] for i in keep_indices]

    # Step 4: Redistribute extras to the nearest other group
    for extra_addr, extra_coord in zip(extras, extras_coords):
        min_dist = float('inf')
        best_group = None
        for idx, coords in enumerate(group_coords):
            # Calculate centroid only if coords list is not empty
            if coords:
                centroid = np.mean(coords, axis=0)
                dist = geodesic(tuple(centroid), extra_coord).miles
                if dist < min_dist:
                    min_dist = dist
                    best_group = idx
        if best_group is not None:
            groups[best_group].append(extra_addr)
            group_coords[best_group].append(extra_coord)

    # Step 5: Optimize and balance routes with batch processing
    status_text.info("Optimizing and balancing routes...")

    # First, optimize each route using Google Directions API
    optimized_groups = []
    route_times = []
    route_stats = []  # List to store route statistics

    # Process routes in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_group = {
            executor.submit(get_google_directions_route, group, GGL_DIRECTIONS_KEY): i
            for i, group in enumerate(groups) if len(group) > 0
        }

        # Initialize results list
        optimized_groups = [[] for _ in range(len(groups))]
        route_times = [0] * len(groups)

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_group):
            group_idx = future_to_group[future]
            try:
                ordered_route, total_time = future.result()
                optimized_groups[group_idx] = ordered_route
                route_times[group_idx] = total_time

                # Store route statistics
                route_stats.append({
                    'route_number': group_idx + 1,
                    'num_addresses': len(ordered_route),
                    'total_time_minutes': round(total_time / 60, 1),
                    'addresses': ordered_route
                })
            except Exception as e:
                st.warning(f"Error optimizing route: {e}")
                optimized_groups[group_idx] = groups[group_idx]
                route_times[group_idx] = 0
                route_stats.append({
                    'route_number': group_idx + 1,
                    'num_addresses': len(groups[group_idx]),
                    'total_time_minutes': 0,
                    'addresses': groups[group_idx]
                })

    # Calculate average route time and identify unbalanced routes
    avg_time = sum(route_times) / len(route_times) if route_times else 0
    max_time_diff = 0.3  # Maximum allowed difference from average (30%)

    # Store address movements for logging
    address_movements = []

    # Attempt to balance routes if they're significantly unbalanced
    if avg_time > 0:
        max_attempts = 3  # Limit the number of balancing attempts
        for attempt in range(max_attempts):
            routes_balanced = True
            # Recalculate average time at the start of each attempt
            avg_time = sum(route_times) / len(route_times) if route_times else 0
            
            if avg_time == 0: # Avoid division by zero
                break

            for i in range(len(route_times)):
                if route_times[i] > avg_time * (1 + max_time_diff):
                    routes_balanced = False
                    # This route is too long, try to move some addresses
                    long_route = optimized_groups[i]
                    if len(long_route) > 3:  # Only attempt to balance if route has enough addresses
                        # Find the shortest route to move addresses to
                        # Ensure shortest route index is not the current route being balanced
                        shortest_route_idx = -1
                        min_time = float('inf')
                        for j in range(len(route_times)):
                            if j != i and route_times[j] < min_time:
                                min_time = route_times[j]
                                shortest_route_idx = j
                                
                        if shortest_route_idx != -1:
                            # Calculate how many addresses to move
                            addresses_to_move = min(
                                len(long_route) - 3,  # Don't go below 3 addresses
                                max(1, len(long_route) // 4)  # Move up to 1/4 of addresses
                            )

                            # Move addresses to the shortest route
                            for _ in range(addresses_to_move):
                                if len(long_route) > 3:  # Keep at least 3 addresses
                                    address_to_move = long_route.pop()
                                    optimized_groups[shortest_route_idx].append(address_to_move)

                                    # Log the address movement
                                    address_movements.append({
                                        'attempt': attempt + 1,
                                        'from_route': i + 1,
                                        'to_route': shortest_route_idx + 1,
                                        'address': address_to_move,
                                        'reason': f"Route {i + 1} was {round((route_times[i] - avg_time) / avg_time * 100)}% longer than average"
                                    })

                            # Re-optimize both routes using parallel processing
                            routes_to_reoptimize = [i, shortest_route_idx]
                            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as reopt_executor:
                                reopt_futures = {
                                    reopt_executor.submit(get_google_directions_route, optimized_groups[idx], GGL_DIRECTIONS_KEY): idx
                                    for idx in routes_to_reoptimize
                                }
                                for reopt_future in concurrent.futures.as_completed(reopt_futures):
                                    reopt_idx = reopt_futures[reopt_future]
                                    try:
                                        ordered_route, total_time = reopt_future.result()
                                        optimized_groups[reopt_idx] = ordered_route
                                        route_times[reopt_idx] = total_time

                                        # Update route statistics
                                        # Find the correct stat entry
                                        stat_entry = next((s for s in route_stats if s['route_number'] == reopt_idx + 1), None)
                                        if stat_entry:
                                             stat_entry['total_time_minutes'] = round(route_times[reopt_idx] / 60, 1)
                                             stat_entry['addresses'] = optimized_groups[reopt_idx]
                                        else: # Add if not found (shouldn't happen with initial population)
                                            route_stats.append({
                                                'route_number': reopt_idx + 1,
                                                'num_addresses': len(optimized_groups[reopt_idx]),
                                                'total_time_minutes': round(route_times[reopt_idx] / 60, 1),
                                                'addresses': optimized_groups[reopt_idx]
                                            })

                                    except Exception as e:
                                        st.warning(f"Error re-optimizing route {reopt_idx + 1}: {e}")

            if routes_balanced:
                break


    # Store route statistics and movements in session state for display
    st.session_state.route_stats = route_stats
    st.session_state.address_movements = address_movements

    # Verify all addresses are accounted for
    all_optimized_addresses = set()
    for group in optimized_groups:
        all_optimized_addresses.update(group)

    if len(all_optimized_addresses) != len(valid_addresses):
        st.warning("Some addresses may have been lost during optimization. Please check the routes carefully.")

    status_text.empty()
    return optimized_groups


def get_google_directions_route(addresses, api_key):
    """
    Get optimal driving route using Google Directions API with real-time traffic data.

    Parameters:
    - addresses: List of addresses to visit
    - api_key: Google Maps API key

    Returns:
    - ordered_addresses: List of addresses in optimal visiting order
    - total_time: Total drive time in seconds (traffic-aware)

    Special Handling:
    - All addresses are formatted to include Minnesota, USA
    - First address is used as both start and end point (round trip)
    - Waypoints are optimized for minimal driving time
    - Uses real-time traffic data for accurate time estimates
    - Considers current traffic conditions and time of day
    """
    if len(addresses) < 2:
        return addresses, 0

    # Ensure all addresses include Minnesota for accurate routing
    formatted_addresses = []
    for addr in addresses:
        if "Minnesota" not in addr and "MN" not in addr:
            formatted_addresses.append(f"{addr}, Minnesota, USA")
        else:
            formatted_addresses.append(addr)

    origin = formatted_addresses[0]
    destination = formatted_addresses[0]  # For a round trip
    waypoints = formatted_addresses[1:]

    # Get current time in seconds since epoch for traffic-aware routing
    current_time = int(time.time())

    # Google Directions API parameters with traffic-aware settings
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "waypoints": "optimize:true|" + "|".join(waypoints),
        "key": api_key,
        "mode": "driving",
        "alternatives": "false",  # Don't need alternative routes
        "departure_time": current_time,  # Use current time for traffic-aware routing
        "traffic_model": "best_guess",  # Use best guess for traffic prediction
        "avoid": "ferries|indoor",  # Avoid ferries and indoor navigation
        "language": "en",
        "units": "imperial"  # Use miles and minutes
    }

    # Add retry logic for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data["status"] != "OK":
                # Log the full error message from the API
                error_message = data.get('error_message', 'No specific error message provided.')
                logger.error(f"Google Directions API request failed (Attempt {attempt + 1}/{max_retries}): Status - {data['status']}, Message - {error_message}")
                raise Exception(f"Google Directions API error: {data['status']} - {error_message}")

            # Get the optimized order of waypoints
            order = data["routes"][0]["waypoint_order"]
            ordered_addresses = [origin] + [waypoints[i] for i in order]

            # Calculate total drive time from all route legs, using duration_in_traffic if available
            total_time = 0
            for leg in data["routes"][0]["legs"]:
                # Use duration_in_traffic if available, otherwise fall back to duration
                if "duration_in_traffic" in leg:
                    total_time += leg["duration_in_traffic"]["value"]
                else:
                    total_time += leg["duration"]["value"]

            # Store traffic information for logging (using ordered addresses as key)
            traffic_info = {
                'has_traffic_data': any('duration_in_traffic' in leg for leg in data["routes"][0]["legs"]),
                'traffic_levels': [leg.get('duration_in_traffic', {}).get('text', 'No traffic data') for leg in data["routes"][0]["legs"]],
                'total_distance': data["routes"][0]["legs"][-1]["distance"]["text"],
                'total_duration': data["routes"][0]["legs"][-1]["duration"]["text"],
                'total_duration_in_traffic': data["routes"][0]["legs"][-1].get("duration_in_traffic", {}).get("text", "No traffic data")
            }

            # Store traffic info in session state for display
            if not hasattr(st.session_state, 'traffic_info_cache'):
                st.session_state.traffic_info_cache = {}
            # Use the tuple of the original addresses as the key for caching, not the ordered ones
            # This assumes the API call for a set of addresses will always return the same traffic info
            # (although the order might differ). A more robust cache might use the parameters themselves as the key.
            original_addresses_tuple = tuple(addresses)
            st.session_state.traffic_info_cache[original_addresses_tuple] = traffic_info


            return ordered_addresses, total_time

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to get directions after {max_retries} attempts: {str(e)}")
                raise e
            logger.warning(f"Retrying Google Directions API call after error: {str(e)}")
            time.sleep(1)  # Wait before retrying

    raise Exception("Failed to get directions after multiple attempts")


# Modify order_route_by_distance to optionally use Google Directions API
def order_route_by_distance(coords, addresses, api_key=None):
    """
    Order addresses in a group to minimize travel from a fixed start point (MidCity Kitchen).
    If api_key is provided, use Google Directions API for drive-time optimization.
    Otherwise, use greedy nearest-neighbor heuristic.
    """
    START_ADDRESS = "693 Raymond Ave, Saint Paul, MN"
    # Ensure START_ADDRESS is not already in addresses to avoid duplicates in the API call
    addresses_unique = [addr for addr in addresses if addr != START_ADDRESS]

    if api_key and addresses_unique:
        # Always start from MidCity Kitchen
        addresses_with_start = [START_ADDRESS] + addresses_unique
        try:
            # Use the main API function for consistency and caching
            ordered_addresses, total_time = get_google_directions_route(addresses_with_start, api_key)
            # Remove the start address from the output if it wasn't in the original list
            if START_ADDRESS not in addresses:
                 return [addr for addr in ordered_addresses if addr != START_ADDRESS]
            else:
                 return ordered_addresses

        except Exception as e:
            st.warning(f"Google Directions API failed for ordering: {e}. Falling back to distance-based ordering.")
            # Fallback to distance-based ordering
    # Fallback: use nearest-neighbor if no API key or API failed
    start_coord = get_coordinates(START_ADDRESS) # Use cached geocoding
    if not coords or len(coords) != len(addresses) or not start_coord:
        # If geocoding failed for the start address or addresses, return original order
        if not start_coord:
             st.warning("Could not geocode start address for distance-based ordering. Returning original order.")
        elif len(coords) != len(addresses):
             st.warning("Coordinate mismatch for distance-based ordering. Returning original order.")
        return addresses # Return original order if cannot order by distance

    visited = [False] * len(coords)
    route = []
    current_coord = start_coord

    # Find the closest address to the start address to begin the greedy route
    min_dist = float('inf')
    next_index = -1
    for i, coord in enumerate(coords):
         dist = geodesic(current_coord, coord).miles
         if dist < min_dist:
              min_dist = dist
              next_index = i

    if next_index != -1:
        visited[next_index] = True
        route.append(addresses[next_index])
        current_coord = coords[next_index]

        # Continue with the rest of the addresses
        for _ in range(len(coords) - 1): # Iterate for the remaining addresses
            min_dist = float('inf')
            next_index = -1
            for i, coord in enumerate(coords):
                if not visited[i]:
                    dist = geodesic(current_coord, coord).miles
                    if dist < min_dist:
                        min_dist = dist
                        next_index = i
            if next_index != -1: # Add address if found
                visited[next_index] = True
                route.append(addresses[next_index])
                current_coord = coords[next_index]
            else: # Break if no unvisited addresses found (shouldn't happen if logic is correct)
                 break

    # If somehow the greedy approach didn't include all addresses, fall back
    if len(route) != len(addresses):
        st.warning("Distance-based ordering did not include all addresses. Falling back to original order.")
        return addresses

    return route


def generate_directions_link(addresses, api_key=GGL_DIRECTIONS_KEY):
    """Generate a Google Maps directions link for a set of addresses in optimized order"""
    if not addresses:
        return "" # Return empty string if no addresses

    START_ADDRESS = "693 Raymond Ave, Saint Paul, MN"
    
    try:
        # Get optimized route order from Google Directions API
        # Include START_ADDRESS in the API call to get the route starting from there
        addresses_with_start = [START_ADDRESS] + [addr for addr in addresses if addr != START_ADDRESS]
        ordered_addresses, _ = get_google_directions_route(addresses_with_start, api_key)

        # Base Google Maps URL
        base_url = "https://www.google.com/maps/dir/"

        # Add each address to the URL in optimized order
        for address in ordered_addresses:
            # Replace spaces with plus signs and append to the URL
            base_url += address.replace(" ", "+") + "/"

        return base_url
    except Exception as e:
        # Fallback to original order and include START_ADDRESS if API call fails
        logger.warning(f"Failed to get optimized order for directions link: {e}. Falling back to original order.")
        base_url = "https://www.google.com/maps/dir/"
        base_url += START_ADDRESS.replace(" ", "+") + "/"
        for address in addresses: # Use original addresses here
            base_url += address.replace(" ", "+") + "/"
        return base_url


def send_email(recipient, subject, body):
    """Send an email using Gmail SMTP."""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = recipient
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Login
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)

        # Send email
        text = msg.as_string()
        server.sendmail(GMAIL_USER, recipient, text)

        # Close session
        server.quit()

        return True
    except Exception as e:
        logger.error(f"Failed to send email to {recipient}: {str(e)}")
        return False


# Functions to handle navigation (within the page)
def go_to_step(step):
    st.session_state.step = step
    # In a multi-page app, rerunning might reset session state managed in the main app.
    # Need to be careful here or manage state differently.
    # For now, assume st.rerun() works as intended within the page context.
    st.rerun()


# Main app UI
def app_home_page():

    # Initialize session state variables
    initialize_session_state()

    st.markdown("<h1 style='text-align: center;'>Generate Routes</h1>", unsafe_allow_html=True)
    st.write("---")

    # Show progress
    st.markdown("### Current Progress")
    progress_percentage = (st.session_state.step / 5) * 100
    st.progress(progress_percentage / 100)
    current_step_name = {
        1: "Select Team Members",
        2: "Upload Route Document",
        3: "Review Routes",
        4: "Assign Routes",
        5: "Confirmation"
    }.get(st.session_state.step, "")
    st.info(f"Step {st.session_state.step}: {current_step_name}")
    st.markdown("---")

    # Step 1: Select emails
    if st.session_state.step == 1:
        st.header("Step 1: Select Team Members")
        st.write("Choose the team members who will be running routes.")

        # Initialize selected_emails in session state if not present
        if 'selected_emails' not in st.session_state:
            st.session_state.selected_emails = []

        # Use a unique key for this specific instance of the multiselect
        selected_names = st.multiselect(
            "Select Team Members:",
            options=list(EMAILS.keys()),
            default=[name for name, email in EMAILS.items() if email in st.session_state.selected_emails],
            placeholder="Select Brightsiders driving a route today",
            key="team_member_selector_step1"
        )

        # Update session state only when Continue is clicked
        if st.button("Continue", use_container_width=True, key=f"step1_continue_{st.session_state.step}"):
            if selected_names:
                st.session_state.selected_emails = [EMAILS[name] for name in selected_names]
                go_to_step(2)

        # Add mobile installation instructions only on step 1
        st.markdown("---")
        with st.expander("📱 Add to Home Screen Instructions"):
            st.markdown("""
            ### iOS Instructions
            1. Open this page in Safari
            2. Tap the Share button (square with up arrow) at the bottom
            3. Scroll down and tap "Add to Home Screen"
            4. Customize the name if desired
            5. Tap "Add" in the top right

            The app will now appear on your home screen with its own icon.
            When opened, it will launch in full-screen mode without Safari's interface elements.
            """)

    # Step 2: Upload PDF
    elif st.session_state.step == 2:
        st.header("Step 2: Upload Route Document")
        selected_names = [name for name, email in EMAILS.items() if email in st.session_state.selected_emails]
        st.write(f"Selected Team Members: {', '.join(selected_names)}")

        uploaded_file = st.file_uploader("Upload a PDF containing addresses:", type="pdf")

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("⬅️ Back", use_container_width=True, key="step2_back"):
                go_to_step(1)

        if uploaded_file is not None:
            st.session_state.pdf_file = uploaded_file
            with cols[1]:
                if st.button("Extract Addresses and Continue", use_container_width=True, key="step2_continue"):
                    try:
                        addresses = extract_addresses_from_pdf(uploaded_file)
                        if addresses:
                            st.session_state.addresses = addresses
                            go_to_step(3)
                        else:
                            st.warning("No addresses could be extracted from the PDF. Please verify the PDF contains properly formatted addresses and try again.")
                    except Exception as e:
                        st.error(f"Error processing PDF file: {str(e)}. Please try a different file.")

    # Step 3: Review and optimize routes
    elif st.session_state.step == 3:
        st.header("Step 3: Review Routes")
        selected_names = [name for name, email in EMAILS.items() if email in st.session_state.selected_emails]
        st.write(f"Selected Team Members: {', '.join(selected_names)}")

        st.subheader("Extracted Addresses")

        # Allow user to edit addresses if needed
        edited_addresses = []
        for i, address in enumerate(st.session_state.addresses):
            edited_address = st.text_input(f"Address {i+1}", value=address, key=f"address_{i}")
            edited_addresses.append(edited_address)

        st.session_state.addresses = edited_addresses

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("⬅️ Back", use_container_width=True, key="step3_back", disabled=st.session_state.is_optimizing):
                go_to_step(2)

        if edited_addresses:
            with cols[1]:
                if st.button("Optimize Routes and Continue", use_container_width=True, key="step3_continue", disabled=st.session_state.is_optimizing):
                    st.session_state.is_optimizing = True
                    st.rerun()

        # Show optimization status if in progress
        if st.session_state.is_optimizing:
            with st.spinner("Optimizing routes... This may take a few minutes."):
                num_groups = len(st.session_state.selected_emails)
                route_groups = optimize_routes(st.session_state.addresses, num_groups)

                # --- Order each group using Google API if available ---
                ordered_route_groups = []
                all_groups_empty = True

                for i, group in enumerate(route_groups):
                    # Geocode each address in the group to get coordinates for ordering
                    group_coords_for_ordering = [get_coordinates(addr) for addr in group] # Use cached version

                    # Filter out None coords before ordering
                    valid_group_addresses = [group[j] for j, coord in enumerate(group_coords_for_ordering) if coord is not None]
                    valid_group_coords = [coord for coord in group_coords_for_ordering if coord is not None]

                    if valid_group_addresses:
                        all_groups_empty = False
                        # Order the route using the selected method
                        ordered_route = order_route_by_distance(valid_group_coords, valid_group_addresses, api_key=GGL_DIRECTIONS_KEY)
                        ordered_route_groups.append(ordered_route)
                    else:
                        ordered_route_groups.append([])

                # Now that groups are optimized and ordered, save to session state and proceed
                if all_groups_empty:
                    st.error("Failed to create optimized routes. No valid addresses found after processing.")
                    st.session_state.is_optimizing = False
                    st.rerun()
                else:
                    # Create initial assignments based on the processed route_groups
                    assignments = {}
                    for i, email in enumerate(st.session_state.selected_emails):
                        if i < len(ordered_route_groups) and ordered_route_groups[i]:
                            assignments[email] = ordered_route_groups[i]

                    st.session_state.route_groups = ordered_route_groups
                    st.session_state.assignments = assignments
                    st.session_state.is_optimizing = False
                    go_to_step(4)

        # Show warning if optimization is in progress
        if st.session_state.is_optimizing:
            st.warning("Please wait for route optimization to complete. This may take a few minutes.")
            st.info("The system is currently processing addresses and calculating optimal routes. Please do not refresh the page or click any buttons.")

    # Step 4: Assign routes
    elif st.session_state.step == 4:
        try:
            st.header("Step 4: Assign Routes")
            st.write("Review and edit route assignments below:")

            # Create a dictionary to track assignments
            new_assignments = {}
            email_to_group = {}

            # Initialize with current assignments
            for email, addresses in st.session_state.assignments.items():
                for i, group in enumerate(st.session_state.route_groups):
                    if set(addresses) == set(group):
                        email_to_group[email] = i

            # Cache traffic info in session state if not already cached
            if not hasattr(st.session_state, 'cached_traffic_info'):
                st.session_state.cached_traffic_info = {}

            # Use the traffic_info_cache populated in get_google_directions_route
            if hasattr(st.session_state, 'traffic_info_cache'):
                for group_idx, group in enumerate(st.session_state.route_groups):
                    try:
                        route_key = tuple(group)
                        if route_key in st.session_state.traffic_info_cache:
                            st.session_state.cached_traffic_info[group_idx] = st.session_state.traffic_info_cache[route_key]
                    except Exception as e:
                        logger.error(f"Error caching traffic info for group {group_idx}: {str(e)}")
                        continue

            # Show the route assignment interface first
            st.subheader("Route Assignments")
            START_ADDRESS = "693 Raymond Ave, Saint Paul, MN"
            
            for group_idx, route_group in enumerate(st.session_state.route_groups):
                try:
                    st.subheader(f"Route Group {group_idx+1}")

                    # Calculate final route statistics for this group
                    if route_group:
                        try:
                            # Add starting point to the route for calculations
                            route_with_start = [START_ADDRESS] + route_group
                            
                            # Get final optimized route order and metrics including start point
                            ordered_addresses, total_time = get_google_directions_route(route_with_start, GGL_DIRECTIONS_KEY)
                            
                            # Get traffic info for the final route
                            route_key = tuple(route_with_start)
                            if route_key in st.session_state.traffic_info_cache:
                                info = st.session_state.traffic_info_cache[route_key]
                                st.write(f"• Base Time: {info.get('total_duration', 'N/A')} | Expected Traffic Time: {info.get('total_duration_in_traffic', 'N/A')} | Total Distance: {info.get('total_distance', 'N/A')} | Addresses: {len(route_group)}")
                            st.markdown("") # Add a small space

                            # Display addresses in optimized order (excluding start point in display)
                            with st.expander("View addresses"):
                                # Skip the first address (start point) in display
                                for i, addr in enumerate(ordered_addresses[1:], 1):
                                    st.write(f"{i}. {addr}")

                        except Exception as e:
                            logger.warning(f"Error calculating final route metrics for group {group_idx}: {str(e)}")
                            st.warning("Could not calculate final route metrics. Showing basic information only.")
                            st.write(f"• Number of Addresses: {len(route_group)}")

                    # Create a selectbox for assigning a team member
                    current_assignee = None
                    for email, idx in email_to_group.items():
                        if idx == group_idx:
                            current_assignee = email

                    current_assignee_name = next((name for name, email in EMAILS.items() if email == current_assignee), None)

                    options = [name for name, email in EMAILS.items() if email in st.session_state.selected_emails]
                    if current_assignee_name in options:
                        default_index = options.index(current_assignee_name)
                    else:
                        default_index = 0

                    assigned_name = st.selectbox(
                        "Assign to:",
                        options=options,
                        index=default_index,
                        key=f"assign_group_{group_idx}"
                    )

                    # Update the email_to_group mapping with the new assignment
                    email_to_group[EMAILS[assigned_name]] = group_idx
                    new_assignments[EMAILS[assigned_name]] = route_group

                    # --- Display map for this route group ---
                    if route_group:
                        try:
                            # Use cached coordinates if available
                            if not hasattr(st.session_state, 'cached_coordinates'):
                                st.session_state.cached_coordinates = {}

                            if group_idx not in st.session_state.cached_coordinates:
                                # Include start point coordinates for map
                                start_coords = get_coordinates(START_ADDRESS)
                                coords_for_map = [start_coords] if start_coords else []
                                coords_for_map.extend([get_coordinates(addr) for addr in route_group])
                                valid_coords_for_map = [coord for coord in coords_for_map if coord is not None]
                                if valid_coords_for_map:
                                    st.session_state.cached_coordinates[group_idx] = valid_coords_for_map
                                else:
                                    st.info("Could not geocode addresses to display map.")

                            if group_idx in st.session_state.cached_coordinates:
                                map_data = pd.DataFrame(st.session_state.cached_coordinates[group_idx], columns=['latitude', 'longitude'])
                                st.map(map_data)
                        except Exception as e:
                            logger.error(f"Error displaying map for group {group_idx}: {str(e)}")
                            st.warning("Could not display map for this route group.")

                    if group_idx < len(st.session_state.route_groups) - 1:
                        st.markdown("---")

                except Exception as e:
                    logger.error(f"Error processing route group {group_idx}: {str(e)}")
                    st.error(f"Error displaying route group {group_idx + 1}. Please try refreshing the page.")
                    continue

            # Check for duplicate assignments
            assignment_counts = {}
            for email in new_assignments.keys():
                assignment_counts[email] = assignment_counts.get(email, 0) + 1

            duplicates = [email for email, count in assignment_counts.items() if count > 1]

            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("⬅️ Back", use_container_width=True, key="step4_back"):
                    go_to_step(3)

            if not duplicates:
                with cols[1]:
                    if st.button("Confirm Assignments and Continue", use_container_width=True, key="step4_continue"):
                        st.session_state.assignments = new_assignments
                        go_to_step(5)
            else:
                st.error(f"The following team members are assigned to multiple routes: {', '.join(duplicates)}. Please fix duplicate assignments.")

        except Exception as e:
            logger.error(f"Critical error in step 4: {str(e)}")
            st.error("An error occurred while loading step 4. Please try refreshing the page or going back to step 3.")
            if st.button("Go Back to Step 3"):
                go_to_step(3)

    # Step 5: Confirmation and email sending
    elif st.session_state.step == 5:
        st.header("Step 5: Confirmation and Email Templates")
        st.write("Review the final assignments and email templates:")

        # Display final assignments and email templates
        for email, addresses in st.session_state.assignments.items():
            assignee_name = next((name for name, e in EMAILS.items() if e == email), "Unknown")
            st.subheader(f"Team Member: {assignee_name}")
            st.write(f"Email: {email}")

            # Display route statistics using same cache structure as Step 4
            START_ADDRESS = "693 Raymond Ave, Saint Paul, MN"
            route_with_start = [START_ADDRESS] + addresses
            route_key = tuple(route_with_start)

            # Use the same cache as Step 4
            if route_key in st.session_state.traffic_info_cache:
                info = st.session_state.traffic_info_cache[route_key]
            else:
                info = {
                    'total_duration': 'N/A',
                    'total_duration_in_traffic': 'N/A',
                    'total_distance': 'N/A'
                }

            st.write("**Route Summary:**")
            st.write(f"• Base Route Time: {info['total_duration']}")
            st.write(f"• Expected Traffic Time: {info['total_duration_in_traffic']}")
            st.write(f"• Total Distance: {info['total_distance']}")
            st.write(f"• Number of Addresses: {len(addresses)}")
            st.markdown("---")

            # Generate directions link for this route using optimized order
            directions_link = generate_directions_link(addresses)

            # Create address list in optimized order
            try:
                # Add starting point for route optimization
                route_with_start = [START_ADDRESS] + addresses
                ordered_addresses, _ = get_google_directions_route(route_with_start, GGL_DIRECTIONS_KEY)
                # Remove start point from display
                ordered_addresses = ordered_addresses[1:]
            except Exception:
                # Fallback to original order if API call fails
                ordered_addresses = addresses

            address_list = ""
            for addr in ordered_addresses:
                address_list += f"- {addr}\n"

            # Create personalized email template
            first_name = assignee_name.split()[0]  # Get first name

            email_template = f"""Hi {first_name},

            **You have been assigned the following route(s):**

            {address_list}

            **Route Summary:**
            • Base Time: {info['total_duration']}
            • Expected Traffic Time: {info['total_duration_in_traffic']}
            • Total Distance: {info['total_distance']}
            • Number of Addresses: {len(addresses)}

            **Instructions to use these addresses in Google Maps:**

            Option 1: Click one of the links below to open your route (addresses are in optimized order):
            • Google Maps: {directions_link}

            Option 2: Copy and paste addresses into Google Maps:
            1. Go to https://www.google.com/maps/
            2. Click on "Directions"
            3. Enter your starting point
            4. Click "Add destination" and enter the first address
            5. Continue adding destinations for each address

            Thank you for your help!
            Brightside Team :)
            """

            # Display the template in a code block with copy button
            st.code(email_template, language="text")

            # Add copy button
            # if st.button(f"Copy Email Template for {assignee_name}", key=f"copy_{email}"):
            #     st.success("Email template copied to clipboard!")

            # Display show addresses expander
            with st.expander("Show addresses"):
                for i, address in enumerate(ordered_addresses):
                    st.write(f"{i+1}. {address}")

            st.markdown("---")

        # Add send all emails button and back button in the same row
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("⬅️ Back", use_container_width=True, key="step5_back"):
                go_to_step(4)

        with cols[1]:
            if st.button("Send Out Emails", use_container_width=True, key="step5_send_emails"):
                st.session_state.show_confirmation = True
                st.rerun()

        # Show confirmation key input if triggered
        if st.session_state.get('show_confirmation', False):
            st.warning("Please enter the confirmation key to send emails")
            confirmation_key = st.text_input("Confirmation Key:", type="password")

            if st.button("Submit", use_container_width=True):
                if confirmation_key == BRIGHTSIDE_CONFIRMATION_KEY:
                    with st.spinner("Sending emails..."):
                        success = True
                        for email, addresses in st.session_state.assignments.items():
                            assignee_name = next((name for name, e in EMAILS.items() if e == email), "Unknown")
                            directions_link = generate_directions_link(addresses)

                            address_list = ""
                            # Need ordered addresses for email body too
                            try:
                                ordered_addresses_email, _ = get_google_directions_route(addresses, GGL_DIRECTIONS_KEY)
                            except Exception:
                                ordered_addresses_email = addresses # Fallback

                            for addr in ordered_addresses_email:
                                address_list += f"- {addr}\n"

                            # Find the route summary info again for the email body
                            assigned_group_idx = None
                            for idx, group in enumerate(st.session_state.route_groups):
                                if set(addresses) == set(group):
                                    assigned_group_idx = idx
                                    break

                            info = {'total_duration': 'N/A', 'total_duration_in_traffic': 'N/A', 'total_distance': 'N/A'} # Default values
                            if assigned_group_idx is not None and hasattr(st.session_state, 'cached_traffic_info'):
                                if assigned_group_idx in st.session_state.cached_traffic_info:
                                    info = st.session_state.cached_traffic_info[assigned_group_idx]

                            email_body = f"""Hi {assignee_name},

                            **You have been assigned the following route(s):**

                            {address_list}

                            **Route Summary:**
                            • Base Route Time: {info['total_duration']}
                            • Expected Traffic Time: {info['total_duration_in_traffic']}
                            • Total Distance: {info['total_distance']}
                            • Number of Addresses: {len(addresses)}

                            **Instructions to use these addresses in Google Maps:**

                            Option 1: Click one of the links below to open your route (addresses are in optimized order):
                            • Web Browser: {directions_link}
                            • iOS Device: {directions_link.replace("https://www.google.com/maps/dir/", "maps://maps.apple.com/?dirflg=d&")}

                            Option 2: Copy and paste addresses into Google Maps:
                            1. Go to https://www.google.com/maps/
                            2. Click on "Directions"
                            3. Enter your starting point
                            4. Click "Add destination" and enter the first address
                            5. Continue adding destinations for each address

                            Thank you for your help!

                            Brightside Team :)
                            """

                            if not send_email(email, "Your Route Assignment", email_body):
                                success = False
                                st.error(f"Failed to send email to {email}")

                        if success:
                            st.success("All emails sent successfully! You can now exit out or refresh your browser session to reset.")
                            if st.button("Exit Out", use_container_width=True, key="step5_reset"):
                                for key in list(st.session_state.keys()):
                                    if key != "step":
                                        del st.session_state[key]
                                st.session_state.step = 1 # Reset step to 1 on exit
                                st.rerun() # Rerun to go back to step 1
                else:
                    st.error("Invalid confirmation key. Please try again.")

# Call the home page function
app_home_page() 
