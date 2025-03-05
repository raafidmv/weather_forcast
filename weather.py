import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import json
from typing import Dict
import requests
from datetime import datetime, timedelta
import re  # Import the regular expression module
import pytz

# Initialize session state for storing chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class CoordinateExtractor:
    def __init__(self):
        try:
            self.api_key = st.secrets["gemini"]["api_key"]
        except KeyError as e:
            raise Exception("API key not found in Streamlit secrets.")
        
        # Initialize LLM for location extraction
        self.location_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-001",  # Updated model name
            temperature=0,
            google_api_key=self.api_key
        )
        
        self.location_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Extract only the location name from this question. 
            Question: {question}
            Return only the location name in JSON format like {{"location": "extracted_location"}}
            """
        )
        
        # Use RunnableSequence for compatibility with newer LangChain versions
        self.location_chain = RunnableSequence(
            self.location_prompt | self.location_llm
        )
        
        # Initialize LLM for coordinate extraction
        self.coord_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-001",  # Updated model name
            temperature=0,
            google_api_key=self.api_key
        )
        
        self.coord_prompt = PromptTemplate(
            input_variables=["location"],
            template="""
            Provide the exact latitude and longitude coordinates for {location}.
            Return only a JSON object in this exact format:
            {{"lat": "LATITUDE", "lon": "LONGITUDE"}}
            Use exactly 4 decimal places. Only return the JSON, no other text.
            """
        )
        
        self.coord_chain = RunnableSequence(
            self.coord_prompt | self.coord_llm
        )
    
    def get_coordinates(self, question: str) -> Dict:
        try:
            # Extract location name
            location_response = self.location_chain.invoke({"question": question})
            
            # Extract JSON using regex
            json_match = re.search(r'\{.*\}', location_response.content)
            if json_match:
                location_json = json_match.group(0)
                location_data = json.loads(location_json)
                location = location_data["location"]
                
                # Extract coordinates for the location
                coord_response = self.coord_chain.invoke({"location": location})
                
                # Extract JSON using regex
                coord_json_match = re.search(r'\{.*\}', coord_response.content)
                if coord_json_match:
                    coord_json = coord_json_match.group(0)
                    return json.loads(coord_json)
                else:
                    return {"error": "No JSON found in coordinate response"}
            else:
                return {"error": "No JSON found in location response"}
            
        except Exception as e:
            return {"error": f"Failed to extract coordinates: {str(e)}"}

class WeatherByCoordinates:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "http://api.openweathermap.org/data/2.5/forecast"  # Add forecast URL

    def get_weather(self, latitude, longitude):
        try:
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params)
            
            response.raise_for_status()
            
            weather_data = response.json()
            
            return weather_data

        except requests.exceptions.RequestException as e:
            return {"error": f"Error fetching weather data: {str(e)}"}
        except KeyError as e:
            return {"error": "Error processing weather data"}

    def get_forecast(self, latitude, longitude):
        try:
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(self.forecast_url, params=params)
            response.raise_for_status()
            forecast_data = response.json()
            return forecast_data

        except requests.exceptions.RequestException as e:
            return {"error": f"Error fetching forecast data: {str(e)}"}
        except KeyError as e:
            return {"error": "Error processing forecast data"}

def display_weather_card(weather_data, lat, lon, timezone_str):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📍 {weather_data['name']}, {weather_data['sys']['country']}")
        st.markdown(f"<div class='big-temp'>{weather_data['main']['temp']}°C</div>", unsafe_allow_html=True)
        st.markdown(f"Feels like: {weather_data['main']['feels_like']}°C")
        st.markdown(f"_{weather_data['weather'][0]['description'].capitalize()}_")
        st.markdown(f"**General Weather**: {weather_data['weather'][0]['description'].capitalize()}")
        st.markdown(f"**Temperature Range**: {weather_data['main']['temp_min']}°C - {weather_data['main']['temp_max']}°C")
    
    with col2:
        st.markdown("### Details")
        st.markdown(f"**Humidity**: {weather_data['main']['humidity']}%")
        st.markdown(f"**Pressure**: {weather_data['main']['pressure']} hPa")
        st.markdown(f"**Wind Speed**: {weather_data['wind']['speed']} m/s")
        st.markdown(f"**Wind Direction**: {weather_data['wind']['deg']}°")

    st.markdown("### 🌅 Sun Times")

    location_timezone = pytz.timezone(timezone_str)

    sunrise_utc = datetime.utcfromtimestamp(weather_data['sys']['sunrise'])
    sunset_utc = datetime.utcfromtimestamp(weather_data['sys']['sunset'])

    sunrise_local = sunrise_utc.replace(tzinfo=pytz.utc).astimezone(location_timezone)
    sunset_local = sunset_utc.replace(tzinfo=pytz.utc).astimezone(location_timezone)
    
    st.markdown(f"**Sunrise**: {sunrise_local.strftime('%I:%M %p %Z%z')}")
    st.markdown(f"**Sunset**: {sunset_local.strftime('%I:%M %p %Z%z')}")

    st.markdown("### 🗺️ Location Details")
    st.markdown(f"**Latitude**: {lat}°")
    st.markdown(f"**Longitude**: {lon}°")

def display_forecast(forecast_data, timezone_str):
    st.markdown("### Forecasted Weather")
    
    location_timezone = pytz.timezone(timezone_str)
    
    # Group forecasts by day
    daily_forecasts = {}
    for forecast in forecast_data['list']:
        forecast_time_utc = datetime.utcfromtimestamp(forecast['dt'])
        forecast_time_local = forecast_time_utc.replace(tzinfo=pytz.utc).astimezone(location_timezone)
        date = forecast_time_local.strftime("%Y-%m-%d")
        
        if date not in daily_forecasts:
            daily_forecasts[date] = []
        daily_forecasts[date].append(forecast)
    
    # Display daily forecasts
    for date, forecasts in daily_forecasts.items():
        st.markdown(f"#### {date}")
        
        col1, col2, col3 = st.columns(3)  # Display 3 forecasts per row
        
        for i, forecast in enumerate(forecasts[:3]):  # Display only the first 3 forecasts
            forecast_time_utc = datetime.utcfromtimestamp(forecast['dt'])
            forecast_time_local = forecast_time_utc.replace(tzinfo=pytz.utc).astimezone(location_timezone)
            
            if i == 0:
                col = col1
            elif i == 1:
                col = col2
            else:
                col = col3
            
            with col:
                st.markdown(f"**Time**: {forecast_time_local.strftime('%I:%M %p %Z%z')}")
                st.markdown(f"**Weather**: {forecast['weather'][0]['description'].capitalize()}")
                st.markdown(f"**Temperature**: {forecast['main']['temp']}°C")

def get_timezone(lat, lon):
    """
    Fetches the timezone for a given latitude and longitude using the TimeZoneDB API.
    """
    url = f"http://api.timezonedb.com/v2.1/get-time-zone?key=4CFZLKN6P2YZ&format=json&by=position&lat={lat}&lng={lon}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['zoneName']
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching timezone: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        st.error(f"Error parsing timezone response: {e}")
        return None

def main():
    st.set_page_config(page_title="Weather App", page_icon="🌤️", layout="wide")
    
    # Add custom CSS for styling
    st.markdown("""
        <style>
        .weather-card {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        .big-temp {
            font-size: 48px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col_input, col_history = st.columns([2, 1])
    
    with col_input:
        st.title("🌤️ Weather Information App")
        
        # Initialize classes
        coord_extractor = CoordinateExtractor()
        
        weather_bot = WeatherByCoordinates("3aa05e76054ecc5912f49e7968a5a805")

        # User input
        question = st.text_input("Ask about weather in any location:", 
                               placeholder="Example: What's the weather in New York?",
                               key="user_input")

        if question:
            with st.spinner("Fetching weather information..."):
                # Get coordinates
                coord_data = coord_extractor.get_coordinates(question)
                
                if "error" in coord_data:
                    st.error(coord_data["error"])
                else:
                    lat = float(coord_data["lat"].strip('"'))
                    lon = float(coord_data["lon"].strip('"'))
                    
                    # Get weather information
                    weather_data = weather_bot.get_weather(lat, lon)
                    
                    if "error" in weather_data:
                        st.error(weather_data["error"])
                    else:
                        # Fetch forecast data
                        forecast_data = weather_bot.get_forecast(lat, lon)
                        if "error" in forecast_data:
                            st.error(forecast_data["error"])
                            forecast_data = None

                        # Get the timezone for the location
                        timezone_str = get_timezone(lat, lon)
                        if not timezone_str:
                            timezone_str = 'UTC'  # Default to UTC if timezone cannot be fetched

                        # Store in session state
                        st.session_state.chat_history.append({
                            "question": question,
                            "weather_data": weather_data,
                            "coordinates": {"lat": lat, "lon": lon},
                            "timezone": timezone_str
                        })
                        
                        # Display current weather
                        display_weather_card(weather_data, lat, lon, timezone_str)
                        
                        # Display forecast if available
                        if forecast_data:
                            display_forecast(forecast_data, timezone_str)

    # Display chat history in the sidebar
    with col_history:
        st.markdown("### 📝 Search History")
        
        if st.button("Clear History"):
            st.session_state.chat_history.clear()
            
        
        for idx, item in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query: {item['question']}", expanded=False):
                display_weather_card(item['weather_data'], 
                                  item['coordinates']['lat'], 
                                  item['coordinates']['lon'],
                                  item['timezone'])

if __name__ == "__main__":
    main()
