import streamlit as st
import requests
import re
import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="Advanced Space Traffic Control System", layout="wide")

# Function to retrieve and parse TLE data from CelesTrak
@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_satellite_data(num_satellites=50):
    """
    Retrieve TLE data for satellites from Heavens-Above without BeautifulSoup.
    """
    # List of example satellite IDs (e.g., ISS, other satellites)
    satellite_ids = [
        25544, 25546, 25547, 25548, 25549, 25550, 25551, 25552, 25553, 25554
        # Add more satellite IDs as needed
    ]
    
    satellites = []
    
    for satellite_id in satellite_ids:
        url = f"https://www.heavens-above.com/PassSummary.aspx?satid={satellite_id}"
        
        try:
            # Fetch the page content
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Regular expression to extract TLE data from the page
            tle_pattern = re.compile(r'<pre>(.*?)</pre>', re.DOTALL)  # Match <pre> tags
            match = tle_pattern.search(response.text)
            
            if match:
                tle_data = match.group(1).strip()
                satellites.append({
                    'satellite_id': satellite_id,
                    'tle': tle_data
                })
            
            # Stop if we have collected enough satellites
            if len(satellites) >= num_satellites:
                break
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for satellite {satellite_id}: {str(e)}")
    
    return satellites[:num_satellites]

# Example usage:
satellite_data = get_satellite_data(num_satellites=10)

# Print the TLE data for each satellite
for data in satellite_data:
    print(f"Satellite ID: {data['satellite_id']}")
    print(f"TLE Data: {data['tle']}\n")

# Function to calculate satellite positions
def calculate_positions(satellites, current_time):
    positions = {}
    for sat in satellites:
        try:
            # Check if both TLE lines are present
            if 'TLE_LINE1' in sat and 'TLE_LINE2' in sat:
                # Assuming your TLE processing function
                position = calculate_position_from_tle(sat['TLE_LINE1'], sat['TLE_LINE2'], current_time)
                positions[sat['OBJECT_NAME']] = position
            else:
                raise ValueError(f"Missing TLE data for satellite: {sat['OBJECT_NAME']}")
        except Exception as e:
            print(f"Error calculating position for {sat['OBJECT_NAME']}: {e}")
    return positions


# Function to check for potential collisions
def check_collisions(positions, threshold=10):
    """
    Check for potential collisions between satellites
    """
    collisions = []
    for i, (name1, pos1) in enumerate(positions):
        for j, (name2, pos2) in enumerate(positions[i+1:], i+1):
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
            if distance < threshold:
                collisions.append((name1, name2, distance))
    return collisions

# Function to create 3D visualization
def create_visualization(positions):
    """
    Create a 3D visualization of Earth and satellite positions
    """
    # Create Earth
    earth = go.Surface(
        z=np.outer(np.sin(np.linspace(0, np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100))),
        x=np.outer(np.cos(np.linspace(0, np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100))),
        y=np.outer(np.sin(np.linspace(0, np.pi, 100)), np.cos(np.linspace(0, 2*np.pi, 100))),
        colorscale=[[0, 'rgb(0, 0, 255)'], [1, 'rgb(0, 255, 255)']],
        showscale=False
    )

    # Create satellite markers
    satellite_markers = go.Scatter3d(
        x=[pos[0] for _, pos in positions],
        y=[pos[1] for _, pos in positions],
        z=[pos[2] for _, pos in positions],
        mode='markers+text',
        text=[name for name, _ in positions],
        marker=dict(size=5, color='red'),
        textposition="top center"
    )

    # Create the 3D plot
    fig = go.Figure(data=[earth, satellite_markers])
    fig.update_layout(
        title="Satellite Positions",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode='data'
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig

# Function to train ML model for collision risk prediction
def train_ml_model(collision_data):
    """
    Train a Random Forest Classifier to predict collision risk
    """
    X = np.array([(d[2], d[3], d[4]) for d in collision_data])
    y = np.array([d[5] for d in collision_data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    st.write(f"ML Model Accuracy: {accuracy:.2f}")

    joblib.dump(model, 'collision_risk_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

# Function to predict collision risk using the trained ML model
def predict_collision_risk(model, scaler, distance, relative_velocity, time_to_closest_approach):
    """
    Predict collision risk using the trained ML model
    """
    features = np.array([[distance, relative_velocity, time_to_closest_approach]])
    scaled_features = scaler.transform(features)
    risk = model.predict_proba(scaled_features)[0][1]
    return risk

# API endpoint for satellite data
@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_satellite_data_api():
    satellites = get_satellite_data(100)
    return [{"name": sat.get('OBJECT_NAME', 'Unknown'), 
             "norad_cat_id": sat.get('NORAD_CAT_ID', 'Unknown'), 
             "tle_line1": sat.get('TLE_LINE1', 'Unknown'), 
             "tle_line2": sat.get('TLE_LINE2', 'Unknown')} 
            for sat in satellites if all(key in sat for key in ['OBJECT_NAME', 'NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2'])]

# API endpoint for current satellite positions
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_satellite_positions_api():
    satellites = get_satellite_data(100)
    positions = calculate_positions(satellites, datetime.utcnow())
    return [{"name": name, "position": {"x": pos[0], "y": pos[1], "z": pos[2]}} for name, pos in positions]

# API endpoint for collision predictions
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_collision_predictions_api(threshold=10):
    satellites = get_satellite_data(100)
    positions = calculate_positions(satellites, datetime.utcnow())
    collisions = check_collisions(positions, threshold)
    return [{"satellite1": sat1, "satellite2": sat2, "distance": distance} for sat1, sat2, distance in collisions]

# Main Streamlit app
def main():
    st.title("Advanced Space Traffic Control System")
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #1E88E5;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: #FFFFFF;
        background-color: #1E1E1E;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for controls
    st.sidebar.title("Controls")
    num_satellites = st.sidebar.slider("Number of Satellites", 10, 100, 50)
    collision_threshold = st.sidebar.slider("Collision Threshold (km)", 1, 50, 10)

    # Initialize session state
    if 'satellites' not in st.session_state:
        st.session_state.satellites = get_satellite_data(num_satellites)

    if 'collision_data' not in st.session_state:
        st.session_state.collision_data = []

    # Train or load ML model
    if os.path.exists('collision_risk_model.joblib'):
        model = joblib.load('collision_risk_model.joblib')
        scaler = joblib.load('scaler.joblib')
    else:
        if len(st.session_state.collision_data) > 100:
            train_ml_model(st.session_state.collision_data)
            model = joblib.load('collision_risk_model.joblib')
            scaler = joblib.load('scaler.joblib')
        else:
            model = None
            scaler = None

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a placeholder for the plot
        plot_placeholder = st.empty()

    with col2:
        # Create a placeholder for collision warnings
        collision_placeholder = st.empty()

    # Get current time
    current_time = datetime.utcnow()

    # Calculate satellite positions
    positions = calculate_positions(st.session_state.satellites, current_time)

    # Check for collisions
    collisions = check_collisions(positions, collision_threshold)

    # Update visualization
    fig = create_visualization(positions)
    plot_placeholder.plotly_chart(fig, use_container_width=True, config={'responsive': True})

    # Display collision warnings
    collision_text = ""
    if collisions:
        collision_text = "Potential collisions detected:\n"
        for sat1, sat2, distance in collisions:
            collision_text += f"- {sat1} and {sat2}: {distance:.2f} km\n"
            
            # Calculate additional features for ML prediction
            relative_velocity = np.random.uniform(0, 10)  # Simulated relative velocity
            time_to_closest_approach = np.random.uniform(0, 3600)  # Simulated time to closest approach

            # Predict collision risk if model is available
            if model and scaler:
                risk = predict_collision_risk(model, scaler, distance, relative_velocity, time_to_closest_approach)
                collision_text += f"  Collision Risk: {risk:.2%}\n"

            # Store collision data for future ML training
            st.session_state.collision_data.append((sat1, sat2, distance, relative_velocity, time_to_closest_approach, int(distance < 5)))

        collision_placeholder.warning(collision_text)
    else:
        collision_placeholder.success("No collisions detected")

    # Display real-time stats
    st.sidebar.markdown("## Real-time Stats")
    st.sidebar.write(f"Total Satellites: {len(st.session_state.satellites)}")
    st.sidebar.write(f"Potential Collisions: {len(collisions)}")
    st.sidebar.write(f"Last Updated: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Refresh satellite data every 15 minutes
    if current_time.minute % 15 == 0 and current_time.second == 0:
        st.session_state.satellites = get_satellite_data(num_satellites)

# Streamlit app with API functionality
def run():
    st.sidebar.title("API")
    api_option = st.sidebar.selectbox("Select API Endpoint", 
                                      ["Satellite Data", "Current Positions", "Collision Predictions"])

    if st.sidebar.button("Get API Data"):
        if api_option == "Satellite Data":
            data = get_satellite_data_api()
        elif api_option == "Current Positions":
            data = get_satellite_positions_api()
        else:
            data = get_collision_predictions_api()

        st.sidebar.json(data)

    main()

if __name__ == "__main__":
    run()