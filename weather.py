import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from typing import Dict
import requests
from datetime import datetime

# Initialize session state for storing chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class CoordinateExtractor:
    def __init__(self):
        self.api_key = st.secrets["gemini"]["api_key"]
        # st.secrets["gemini"]["api_key"]
        
        self.location_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
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
        
        self.location_chain = LLMChain(llm=self.location_llm, prompt=self.location_prompt)
        
        self.coord_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
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
        
        self.coord_chain = LLMChain(llm=self.coord_llm, prompt=self.coord_prompt)
    
    def get_coordinates(self, question: str) -> Dict:
        try:
            location_response = self.location_chain.run(question=question)
            location_data = json.loads(location_response.strip())
            location = location_data["location"]
            
            coord_response = self.coord_chain.run(location=location)
            return json.loads(coord_response.strip())
            
        except Exception as e:
            return {"error": f"Failed to extract coordinates: {str(e)}"}

class WeatherByCoordinates:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

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
        except KeyError:
            return {"error": "Error processing weather data"}

def display_weather_card(weather_data, lat, lon):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📍 {weather_data['name']}, {weather_data['sys']['country']}")
        st.markdown(f"<div class='big-temp'>{weather_data['main']['temp']}°C</div>", unsafe_allow_html=True)
        st.markdown(f"Feels like: {weather_data['main']['feels_like']}°C")
        st.markdown(f"_{weather_data['weather'][0]['description'].capitalize()}_")
    
    with col2:
        st.markdown("### Details")
        st.markdown(f"Humidity: {weather_data['main']['humidity']}%")
        st.markdown(f"Pressure: {weather_data['main']['pressure']} hPa")
        st.markdown(f"Wind Speed: {weather_data['wind']['speed']} m/s")
    
    st.markdown("### 🌅 Sun Times")
    col3, col4 = st.columns(2)
    
    with col3:
        # Add 5 hours and 30 minutes to sunrise time
        sunrise = datetime.fromtimestamp(weather_data['sys']['sunrise'])
        sunrise_adjusted = sunrise.replace(
            hour=(sunrise.hour + 5) % 24,
            minute=(sunrise.minute + 30) % 60
        )
        # If minutes overflow, add an hour
        if sunrise.minute + 30 >= 60:
            sunrise_adjusted = sunrise_adjusted.replace(hour=(sunrise_adjusted.hour + 1) % 24)
        st.markdown(f"Sunrise: {sunrise_adjusted.strftime('%I:%M %p')}")
    
    with col4:
        # Add 5 hours and 30 minutes to sunset time
        sunset = datetime.fromtimestamp(weather_data['sys']['sunset'])
        sunset_adjusted = sunset.replace(
            hour=(sunset.hour + 5) % 24,
            minute=(sunset.minute + 30) % 60
        )
        # If minutes overflow, add an hour
        if sunset.minute + 30 >= 60:
            sunset_adjusted = sunset_adjusted.replace(hour=(sunset_adjusted.hour + 1) % 24)
        st.markdown(f"Sunset: {sunset_adjusted.strftime('%I:%M %p')}")
    
    st.markdown("### 🗺️ Location Details")
    st.markdown(f"Latitude: {lat}°")
    st.markdown(f"Longitude: {lon}°")


def main():
    st.set_page_config(page_title="Weather App", page_icon="🌤️", layout="wide")
    
    # Add custom CSS
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
        .weather-detail {
            margin: 5px 0;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .chat-message {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .system-message {
            background-color: #f5f5f5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col_input, col_history = st.columns([2, 1])
    
    with col_input:
        st.title("🌤️ Weather Information App")
        
        # Initialize classes
        coord_extractor = CoordinateExtractor()
        weather_bot = WeatherByCoordinates(st.secrets["weather"]["api_key"])

        # User input
        question = st.text_input("Ask about weather in any location:", 
                               placeholder="Example: What's the weather in New York?",
                               key="user_input")

        if question:
            with st.spinner("Fetching weather information..."):
                # Get coordinates
                coord_data = coord_extractor.get_coordinates(question)
                
                if "error" in coord_data:
                    st.error(f"Error: {coord_data['error']}")
                else:
                    # Remove quotes from coordinates if present
                    lat = float(coord_data["lat"].strip('"'))
                    lon = float(coord_data["lon"].strip('"'))
                    
                    # Get weather information
                    weather_data = weather_bot.get_weather(lat, lon)
                    
                    if "error" in weather_data:
                        st.error(weather_data["error"])
                    else:
                        # Store in session state
                        st.session_state.chat_history.append({
                            "question": question,
                            "weather_data": weather_data,
                            "coordinates": {"lat": lat, "lon": lon}
                        })
                        
                        # Display current weather
                        display_weather_card(weather_data, lat, lon)

    # Display chat history in the sidebar
    with col_history:
        st.markdown("### 📝 Search History")
        if st.button("Clear History"):
            st.session_state.chat_history = []
            # st.rerun()

            
        for idx, item in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query: {item['question']}", expanded=False):
                display_weather_card(item['weather_data'], 
                                  item['coordinates']['lat'], 
                                  item['coordinates']['lon'])

if __name__ == "__main__":
    main()
