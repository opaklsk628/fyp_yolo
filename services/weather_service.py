import requests

class WeatherService:
    def get_weather(self):
        """Get weather data from Hong Kong Observatory"""
        try:
            forecast_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en"
            forecast_response = requests.get(forecast_url, timeout=5)
            forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
            
            current_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
            current_response = requests.get(current_url, timeout=5)
            current_data = current_response.json() if current_response.status_code == 200 else None
            
            warning_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en"
            warning_response = requests.get(warning_url, timeout=5)
            warning_data = warning_response.json() if warning_response.status_code == 200 else None
            
            weather_info = {
                'current': None,
                'forecast': [],
                'warnings': []
            }
            
            if current_data and 'temperature' in current_data:
                temps = current_data['temperature']['data']
                if temps:
                    avg_temp = sum(t['value'] for t in temps) / len(temps)
                    weather_info['current'] = {
                        'temperature': round(avg_temp, 1),
                        'humidity': current_data.get('humidity', {}).get('data', [{}])[0].get('value', 'N/A')
                    }
            
            if forecast_data and 'weatherForecast' in forecast_data:
                for day in forecast_data['weatherForecast'][:7]:
                    weather_info['forecast'].append({
                        'date': day.get('forecastDate', ''),
                        'week': day.get('week', ''),
                        'min_temp': day.get('forecastMintemp', {}).get('value', 'N/A'),
                        'max_temp': day.get('forecastMaxtemp', {}).get('value', 'N/A'),
                        'weather': day.get('forecastWeather', '')
                    })
            
            if warning_data:
                for warning_type, warning_info in warning_data.items():
                    if warning_info and isinstance(warning_info, dict) and warning_info.get('name'):
                        weather_info['warnings'].append({
                            'type': warning_type,
                            'name': warning_info.get('name'),
                            'code': warning_info.get('code', '')
                        })
            
            return weather_info
            
        except Exception as e:
            print(f"Error fetching HKO weather: {e}")
            return None
