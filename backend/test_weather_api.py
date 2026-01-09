"""Quick test for weather service"""
from integrations.weather_service import get_weather_service

print("🧪 Testing Weather Service...")

# Initialize service
weather_service = get_weather_service()

# Test current weather
print("\n📊 Current Weather:")
current = weather_service.get_current_weather()
for key, value in current.items():
    print(f"  {key}: {value}")

# Test hazards
print("\n⚠️ Weather Hazards:")
hazards = weather_service.get_weather_hazards()
print(f"  Found {len(hazards)} hazards")
for hazard in hazards:
    print(f"  - {hazard['type']}: {hazard['description']}")

# Test forecast
print("\n🔮 3-Day Forecast:")
forecast = weather_service.get_weather_forecast(days=3)
for day in forecast[:3]:
    print(f"  {day['date']}: {day['temperature']}°C, {day['condition']}")

print("\n✅ Weather service test complete!")