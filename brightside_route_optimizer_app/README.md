# Brightside Route Optimizer

A Streamlit application that optimizes delivery routes for Brightside's Fresh Produce PWYC (Pay What You Can) service in the Minnesota metro area, using a combination of machine learning clustering and real-time traffic data to create balanced, efficient routes based on number of team members volunterring to drive delivery routes that day.

## Overview

The Brightside Route Optimizer is designed to efficiently distribute delivery addresses among team members while considering:
- Geographic proximity
- Real-time traffic conditions
- Special delivery requirements
- Route balancing
- Time of day

## How It Works

### Two-Phase Optimization Approach

The application uses a two-phase approach to optimize routes:

1. **Phase 1: Geographic Clustering (K-means)**
   - Uses latitude/longitude coordinates to create initial groups
   - Groups addresses that are geographically close to each other
   - Provides a quick first pass to get reasonable starting groups
   - Special case handling for specific addresses (e.g., '1920 4th Ave S')

2. **Phase 2: Drive-Time Optimization (Google Maps API)**
   - Takes the groups from Phase 1
   - Uses Google Directions API to find optimal driving order
   - Calculates actual drive times between addresses
   - Optimizes routes to minimize total drive time
   - Incorporates real-time traffic data

### Machine Learning Components

#### K-means Clustering
- **Purpose**: Initial geographic grouping of addresses
- **Parameters**:
  - `n_clusters`: Number of route groups (one per driver)
  - `random_state`: Fixed for reproducibility
  - `n_init`: 5 (reduced from 10 for speed)
  - `max_iter`: 100 (limited for performance)
  - `algorithm`: 'elkan' (faster for dense data)

#### Route Balancing
- **Purpose**: Ensure fair distribution of work
- **Process**:
  1. Calculate average drive time across all routes
  2. Identify routes >30% longer than average
  3. Move addresses from long routes to shorter ones
  4. Re-optimize affected routes
  5. Repeat up to 3 times if needed

### Special Cases and Constraints

1. **1920 4th Ave S Rule**
   - Routes containing this address are limited to 3 addresses
   - Extra addresses are redistributed to nearest other group
   - Ensures manageable workload for bag distribution

2. **Size Protection**
   - Minimum route size: 3 addresses
   - Maximum addresses moved per attempt: min(route_size - 3, route_size // 4)
   - Prevents routes from becoming too small or too large

3. **Traffic-Aware Routing**
   - Uses real-time traffic data from Google Maps API
   - Considers current traffic conditions and time of day
   - Provides both base duration and traffic-affected duration

## Implementation Details

### Performance Optimizations

1. **Parallel Processing**
   - Uses ThreadPoolExecutor for parallel geocoding
   - Processes multiple addresses simultaneously
   - 5 worker threads for geocoding
   - 3 worker threads for route optimization

2. **Caching**
   - LRU cache for geocoding results
   - Prevents repeated API calls for same address
   - Cache size: 1000 addresses

3. **API Optimization**
   - Retry logic for API calls (3 attempts)
   - 10-second timeout
   - Traffic-aware routing parameters
   - Batch processing where possible

### Known Limitations and Trade-offs

1. **Geocoding Accuracy**
   - Relies on Nominatim geocoding service
   - May have occasional timeouts or inaccuracies
   - Added "Minnesota, USA" to addresses for better accuracy
   - 1.5-second delay between requests to respect rate limits

2. **Route Balancing**
   - 30% threshold for route balancing
   - May not achieve perfect balance due to:
     - Geographic constraints
     - Special case rules
     - Traffic variations
   - Limited to 3 balancing attempts to prevent infinite loops

3. **Traffic Data**
   - Traffic data availability depends on:
     - Time of day
     - Location
     - Google Maps API coverage
   - Falls back to base duration if traffic data unavailable

4. **Performance vs. Accuracy**
   - Reduced K-means iterations for speed
   - Limited balancing attempts
   - Trade-off between optimization time and route quality

## Setup and Configuration

1. **Required API Keys and Credentials**
   - Google Maps Directions API key
   - Brightside confirmation key
   - Gmail credentials for sending emails
   - AWS credentials for S3 access

2. **Configuration**
   Create a `.streamlit/secrets.toml` file in the project root with the following structure:
   ```toml
   # Google Maps API Key
   GGL_DIRECTIONS_KEY = "your_google_maps_api_key_here"

   # Brightside Confirmation Key
   BRIGHTSIDE_CONFIRMATION_KEY = "your_confirmation_key_here"

   # Gmail Configuration for Sending Emails
   GMAIL_USER = "your_gmail_address@gmail.com"
   GMAIL_APP_PASSWORD = "your_gmail_app_password"

   # AWS Configuration
   aws_access_key_id = "your_access_key_id"
   aws_secret_access_key = "your_secret_access_key"
   aws_region = "your_region"  # e.g., "us-east-1"
   ```

3. **Theme Configuration**
   The app uses a custom theme defined in `.streamlit/config.toml`:
   ```toml
   [theme]
   primaryColor = "#0066cc"      # Main accent color for buttons and interactive elements
   backgroundColor = "#ffffff"    # Main background color of the app
   secondaryBackgroundColor = "#f0f2f6"  # Background color for sidebar and containers
   textColor = "#262730"         # Color of text throughout the app
   font = "sans serif"           # Font family for all text

   [server]
   enableCORS = false            # Prevents other websites from making requests to this app
   enableXsrfProtection = true   # Adds security tokens to forms to prevent unauthorized access
   ```

4. **Running the App**
   To run the app with the correct configuration:
   ```bash
   cd brightside_route_optimizer_app
   streamlit run app_main.py
   ```
   
   Note: The app must be run from within the `brightside_route_optimizer_app` directory for the configuration files to be loaded properly.

5. **Team Members Data Setup**
   Create a CSV file named `team_members.csv` with the following structure:
   ```csv
   name,email
   e.g. "John Doe","JohnDoe@gmail.com"
   ```

   You can generate this CSV file using the provided script:
   ```bash
   python create_team_members_csv.py
   ```

   **Amazon S3 Setup**:
   1. **Create S3 Bucket**
      - Log into AWS Console
      - Navigate to S3 service
      - Click "Create bucket"
      - Name the bucket `bucket-name-here`
      - Choose your preferred region
      - Keep default settings for other options
      - Click "Create bucket"

   2. **Upload CSV File**
      - Select your newly created bucket
      - Click "Upload"
      - Drag and drop or select the `csv-name.csv` file
      - Click "Upload"

   3. **Configure Bucket Permissions**
      - Select your bucket
      - Go to "Permissions" tab
      - Under "Block public access":
        * Click "Edit"
        * Uncheck "Block all public access"
        * Save changes
      - Under "Bucket policy":
        * Click "Edit"
        * Add a policy that allows read access for your application
        * Save changes

   4. **Create IAM User for Application**
      - Go to IAM service in AWS Console
      - Click "Users" → "Add user"
      - Create a username (e.g., "brightside-app")
      - Select "Programmatic access"
      - Attach the "AmazonS3ReadOnlyAccess" policy
      - Complete user creation
      - Save the Access Key ID and Secret Access Key
      - Add these credentials to your `secrets.toml` file:
        ```toml
        aws_access_key_id = "your_access_key_id"
        aws_secret_access_key = "your_secret_access_key"
        aws_region = "your_region"  # e.g., "us-east-1"
        ```

   **Note**: The app will automatically load team members from this CSV file on startup.

6. **Email Sending Features**
   - The app can send personalized route assignments to team members
   - Each email includes:
     - Personalized greeting with team member's name
     - List of assigned addresses
     - Google Maps directions link
     - Step-by-step instructions for using the route
   - Emails are sent using Gmail SMTP
   - Requires confirmation key before sending
   - Provides copy functionality for manual sending if needed

## Local Development Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation Steps

1. **Clone the Repository**
```bash
git clone <repository-url>
cd brightside_route_optimizer_app
```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Up API Keys**
   Create a `.streamlit/secrets.toml` file in the project root with the following structure:
   ```toml
   # Google Maps API Key
   GGL_DIRECTIONS_KEY = "your_google_maps_api_key_here" - your google directions api key

   # Brightside Confirmation Key
   BRIGHTSIDE_CONFIRMATION_KEY = "your_confirmation_key_here" - this is used to send the emails
   ```

   To obtain the required API keys:
   - **Google Maps API Key**:
     1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
     2. Create a new project or select an existing one
     3. Enable the "Directions API" and "Maps JavaScript API"
     4. Create credentials (API key)
     5. Restrict the API key to only the required APIs
     6. Copy the API key to your `secrets.toml` file

   - **Brightside Confirmation Key**:
     - This is a custom key used to verify email sending
     - Can create your own key

5. **Run the Application**
Run this in terminal
```bash
cd brightside_route_optimizer_app && streamlit run app_main.py
```
   The application will be available at `http://localhost:8501`

### Development Notes

1. **API Key Security**
   - Never commit API keys to version control
   - Keep your `secrets.toml` file secure
   - Use environment variables in production

2. **Rate Limits**
   - Google Maps API has usage limits
   - Nominatim geocoding service has rate limits
   - The app includes built-in delays to respect these limits

3. **Testing**
   - Test with a small set of addresses first
   - Monitor API usage in Google Cloud Console
   - Check geocoding accuracy with known addresses

4. **Troubleshooting**
   - If geocoding fails, check address format
   - If Google Maps API fails, verify API key and quotas
   - Check logs for detailed error messages

## Usage

1. Select team members
2. Upload route document (PDF)
3. Review and edit addresses if needed
4. Review optimized routes and assignments
5. Confirm and send emails

## Potential Future Improvements

1. **Enhanced Traffic Handling**
   - Consider time windows for deliveries
   - Account for rush hour patterns
   - Predict traffic based on historical data

2. **Advanced Optimization**
   - Implement more sophisticated clustering algorithms
   - Add support for time windows
   - Consider driver preferences and constraints

## License

This project is licensed under the MIT License - see the LICENSE.md file for details. 