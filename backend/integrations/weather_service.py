"""
Weather Service Integration
Connects to OpenWeatherMap API for real weather data
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherService:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather service with API key from .env or parameter"""
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"

        # Fallback to simulated data if no API key
        self.use_simulation = not self.api_key or self.api_key == 'your_api_key_here'

        # San Francisco coordinates
        self.city_coords = {"lat": 37.7749, "lon": -122.4194}

        print(f"🌤️ Weather service initialized (simulation: {self.use_simulation})")

    def get_current_weather(self) -> Dict:
        """Get current weather conditions"""
        if self.use_simulation:
            return self._simulate_current_weather()

        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": self.city_coords["lat"],
                "lon": self.city_coords["lon"],
                "appid": self.api_key,
                "units": "metric"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                print(f"⚠️ Weather API error: {response.status_code}")
                return self._simulate_current_weather()

        except Exception as e:
            print(f"❌ Weather service error: {e}")
            return self._simulate_current_weather()

    def get_weather_hazards(self) -> List[Dict]:
        """Generate hazards based on current weather conditions"""
        weather = self.get_current_weather()
        hazards = []

        # Rain-based flood risks
        if weather['precipitation'] > 5:  # >5mm rain
            severity = 'high' if weather['precipitation'] > 15 else 'medium'

            # Generate multiple flood risk points
            for i in range(random.randint(2, 5)):
                hazards.append({
                    'type': 'flood_risk',
                    'severity': severity,
                    'location': self._generate_nearby_location(),
                    'description': f"Heavy rain ({weather['precipitation']}mm/h) - Flood risk area",
                    'source': 'weather_api',
                    'expires_at': (datetime.now() + timedelta(hours=3)).isoformat()
                })

        # Temperature-based hazards
        if weather['temperature'] < 2:  # Near freezing
            hazards.append({
                'type': 'ice_warning',
                'severity': 'medium',
                'location': self._generate_nearby_location(),
                'description': f"Low temperature ({weather['temperature']}°C) - Ice formation risk",
                'source': 'weather_api',
                'expires_at': (datetime.now() + timedelta(hours=6)).isoformat()
            })

        # Wind-based hazards
        if weather['wind_speed'] > 15:  # >15 m/s strong wind
            hazards.append({
                'type': 'wind_warning',
                'severity': 'medium',
                'location': self._generate_nearby_location(),
                'description': f"Strong winds ({weather['wind_speed']} m/s) - Debris risk",
                'source': 'weather_api',
                'expires_at': (datetime.now() + timedelta(hours=2)).isoformat()
            })

        return hazards

    def get_weather_forecast(self, days: int = 3) -> List[Dict]:
        """Get weather forecast for hazard prediction"""
        if self.use_simulation:
            return self._simulate_forecast(days)

        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": self.city_coords["lat"],
                "lon": self.city_coords["lon"],
                "appid": self.api_key,
                "units": "metric",
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast_data(data)
            else:
                return self._simulate_forecast(days)

        except Exception as e:
            print(f"❌ Forecast error: {e}")
            return self._simulate_forecast(days)

    def _simulate_current_weather(self) -> Dict:
        """Simulate current weather for demo"""
        # Create realistic weather patterns
        weather_patterns = [
            {'temp': 18, 'rain': 0, 'wind': 5, 'condition': 'clear'},
            {'temp': 15, 'rain': 2, 'wind': 8, 'condition': 'light rain'},
            {'temp': 12, 'rain': 8, 'wind': 12, 'condition': 'rain'},
            {'temp': 10, 'rain': 18, 'wind': 15, 'condition': 'heavy rain'},
            {'temp': 22, 'rain': 0, 'wind': 3, 'condition': 'sunny'},
        ]

        pattern = random.choice(weather_patterns)

        return {
            'temperature': round(pattern['temp'] + random.uniform(-3, 3), 2),
            'precipitation': round(max(0, pattern['rain'] + random.uniform(-1, 2)), 1),
            'wind_speed': round(max(0, pattern['wind'] + random.uniform(-2, 3)), 1),
            'humidity': random.randint(40, 90),
            'pressure': random.randint(1010, 1025),
            'condition': pattern['condition'],
            'timestamp': datetime.now().isoformat(),
            'source': 'simulated'
        }

    def _simulate_forecast(self, days: int) -> List[Dict]:
        """Simulate weather forecast"""
        forecast = []
        for day in range(days):
            date = datetime.now() + timedelta(days=day)

            base_temp = 16 + random.uniform(-5, 8)
            rain_chance = random.random()

            forecast.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': round(base_temp, 1),
                'precipitation': round(random.uniform(0, 10) if rain_chance > 0.7 else 0, 1),
                'wind_speed': round(random.uniform(3, 12), 1),
                'condition': 'rain' if rain_chance > 0.7 else 'clear'
            })

        return forecast

    def _generate_nearby_location(self) -> Dict:
        """Generate location near city center"""
        return {
            'lat': self.city_coords['lat'] + random.uniform(-0.05, 0.05),
            'lon': self.city_coords['lon'] + random.uniform(-0.05, 0.05)
        }

    def _parse_weather_data(self, data: Dict) -> Dict:
        """Parse OpenWeatherMap API response"""
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        rain = data.get('rain', {})

        return {
            'temperature': round(main.get('temp', 0), 2),
            'precipitation': round(rain.get('1h', 0), 1),
            'wind_speed': round(wind.get('speed', 0), 1),
            'humidity': main.get('humidity', 0),
            'pressure': main.get('pressure', 0),
            'condition': weather.get('description', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'source': 'openweather'
        }

    def _parse_forecast_data(self, data: Dict) -> List[Dict]:
        """Parse forecast API response"""
        forecasts = []
        for item in data.get('list', []):
            main = item.get('main', {})
            weather = item.get('weather', [{}])[0]
            wind = item.get('wind', {})
            rain = item.get('rain', {})

            forecasts.append({
                'date': datetime.fromtimestamp(item.get('dt', 0)).strftime('%Y-%m-%d'),
                'temperature': round(main.get('temp', 0), 1),
                'precipitation': round(rain.get('3h', 0), 1),
                'wind_speed': round(wind.get('speed', 0), 1),
                'condition': weather.get('description', 'unknown')
            })

        return forecasts

# Global weather service instance
weather_service = None

def get_weather_service(api_key: Optional[str] = None):
    """Get or create weather service instance"""
    global weather_service
    if weather_service is None:
        weather_service = WeatherService(api_key)
    return weather_service