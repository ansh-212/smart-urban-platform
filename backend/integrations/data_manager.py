"""
Real Data Integration Manager
Coordinates all external data sources and updates the system
"""
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import time
import schedule

from .weather_service import get_weather_service
from .construction_service import get_construction_service
from .traffic_service import get_traffic_service

class RealDataManager:
    def __init__(self, socketio, event_manager, db_session_factory):
        self.socketio = socketio
        self.event_manager = event_manager
        self.db_session_factory = db_session_factory

        # Initialize services
        self.weather_service = get_weather_service()
        self.construction_service = get_construction_service()
        self.traffic_service = get_traffic_service()

        # Data update intervals
        self.weather_update_interval = 15  # minutes
        self.traffic_update_interval = 5   # minutes
        self.construction_update_interval = 60  # minutes

        # Background update thread
        self.update_thread = None
        self.running = False

        print("🌍 Real Data Manager initialized")

    def start_real_time_updates(self):
        """Start background data updates"""
        if self.running:
            return

        self.running = True

        # Schedule periodic updates
        schedule.every(self.weather_update_interval).minutes.do(self._update_weather_data)
        schedule.every(self.traffic_update_interval).minutes.do(self._update_traffic_data)
        schedule.every(self.construction_update_interval).minutes.do(self._update_construction_data)

        # Start scheduler thread
        self.update_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.update_thread.start()

        # Initial data fetch
        self._update_all_data()

        print("🔄 Real-time data updates started")

    def stop_real_time_updates(self):
        """Stop background updates"""
        self.running = False
        schedule.clear()
        print("⏸️ Real-time data updates stopped")

    def get_current_external_data(self) -> Dict:
        """Get all current external data"""
        return {
            'weather': self.weather_service.get_current_weather(),
            'weather_hazards': self.weather_service.get_weather_hazards(),
            'weather_forecast': self.weather_service.get_weather_forecast(),
            'construction_zones': self.construction_service.get_construction_zones(),
            'road_closures': self.construction_service.get_road_closures(),
            'traffic_incidents': self.traffic_service.get_traffic_incidents(),
            'traffic_flow': self.traffic_service.get_traffic_flow(),
            'congestion_hotspots': self.traffic_service.get_congestion_hotspots(),
            'traffic_predictions': self.traffic_service.get_predicted_traffic(),
            'last_updated': datetime.now().isoformat()
        }

    def _update_all_data(self):
        """Update all data sources"""
        try:
            print("🔄 Updating all external data sources...")

            # Get all external data
            external_data = self.get_current_external_data()

            # Convert weather hazards to system hazards
            weather_hazards = self._process_weather_hazards(external_data['weather_hazards'])

            # Convert construction zones to system hazards
            construction_hazards = self._process_construction_data(external_data['construction_zones'])

            # Convert traffic incidents to system hazards
            traffic_hazards = self._process_traffic_incidents(external_data['traffic_incidents'])

            # Broadcast external data updates
            self.event_manager.socketio.emit('external_data_update', {
                'type': 'external_data_update',
                'weather': external_data['weather'],
                'traffic_summary': self._summarize_traffic(external_data),
                'total_external_hazards': len(weather_hazards + construction_hazards + traffic_hazards),
                'timestamp': datetime.now().isoformat()
            })

            print(f"✅ External data updated: {len(weather_hazards)} weather + {len(construction_hazards)} construction + {len(traffic_hazards)} traffic hazards")

        except Exception as e:
            print(f"❌ External data update error: {e}")

    def _update_weather_data(self):
        """Update weather data"""
        try:
            weather = self.weather_service.get_current_weather()
            hazards = self.weather_service.get_weather_hazards()

            # Broadcast weather update
            self.event_manager.socketio.emit('weather_update', {
                'type': 'weather_update',
                'weather': weather,
                'hazards_count': len(hazards),
                'timestamp': datetime.now().isoformat()
            })

            print(f"🌤️ Weather updated: {weather['condition']}, {len(hazards)} hazards")

        except Exception as e:
            print(f"❌ Weather update error: {e}")

    def _update_traffic_data(self):
        """Update traffic data"""
        try:
            incidents = self.traffic_service.get_traffic_incidents()
            flow = self.traffic_service.get_traffic_flow()
            hotspots = self.traffic_service.get_congestion_hotspots()

            # Broadcast traffic update
            self.event_manager.socketio.emit('traffic_update', {
                'type': 'traffic_update',
                'incidents': len(incidents),
                'congestion_hotspots': len(hotspots),
                'average_speed': sum(f['current_speed'] for f in flow) / len(flow) if flow else 0,
                'timestamp': datetime.now().isoformat()
            })

            print(f"🚗 Traffic updated: {len(incidents)} incidents, {len(hotspots)} hotspots")

        except Exception as e:
            print(f"❌ Traffic update error: {e}")

    def _update_construction_data(self):
        """Update construction data"""
        try:
            zones = self.construction_service.get_construction_zones()
            closures = self.construction_service.get_road_closures()

            # Broadcast construction update
            self.event_manager.socketio.emit('construction_update', {
                'type': 'construction_update',
                'construction_zones': len(zones),
                'road_closures': len(closures),
                'timestamp': datetime.now().isoformat()
            })

            print(f"🚧 Construction updated: {len(zones)} zones, {len(closures)} closures")

        except Exception as e:
            print(f"❌ Construction update error: {e}")

    def _process_weather_hazards(self, weather_hazards: List[Dict]) -> List[Dict]:
        """Convert weather data to system hazards"""
        processed = []
        for hazard in weather_hazards:
            processed.append({
                'type': hazard['type'],
                'severity': hazard['severity'],
                'location': hazard['location'],
                'description': hazard['description'],
                'source': 'weather_api',
                'external_id': f"weather_{hash(hazard['description'])}",
                'expires_at': hazard.get('expires_at')
            })
        return processed

    def _process_construction_data(self, construction_data: List[Dict]) -> List[Dict]:
        """Convert construction data to system hazards"""
        processed = []
        for item in construction_data:
            processed.append({
                'type': 'construction',
                'severity': item.get('severity', 'medium'),
                'location': item['location'],
                'description': item['description'],
                'source': 'construction_api',
                'external_id': f"construction_{item.get('osm_id', hash(item['description']))}"
            })
        return processed

    def _process_traffic_incidents(self, traffic_incidents: List[Dict]) -> List[Dict]:
        """Convert traffic incidents to system hazards"""
        processed = []
        for incident in traffic_incidents:
            processed.append({
                'type': incident['incident_type'],
                'severity': incident['severity'],
                'location': incident['location'],
                'description': f"{incident['description']} on {incident['road_name']}",
                'source': 'traffic_api',
                'external_id': f"traffic_{hash(incident['description'])}",
                'estimated_clearance': incident.get('estimated_clearance')
            })
        return processed

    def _summarize_traffic(self, external_data: Dict) -> Dict:
        """Create traffic summary"""
        flow = external_data.get('traffic_flow', [])
        incidents = external_data.get('traffic_incidents', [])
        hotspots = external_data.get('congestion_hotspots', [])

        if not flow:
            return {'status': 'no_data'}

        avg_speed = sum(f['current_speed'] for f in flow) / len(flow)
        avg_congestion = sum(f['congestion_level'] for f in flow) / len(flow)

        if avg_congestion > 0.7:
            status = 'heavy'
        elif avg_congestion > 0.4:
            status = 'moderate'
        else:
            status = 'light'

        return {
            'status': status,
            'average_speed': round(avg_speed, 1),
            'congestion_level': round(avg_congestion, 2),
            'incidents': len(incidents),
            'hotspots': len(hotspots)
        }

    def _run_scheduler(self):
        """Run the background scheduler with proper error handling"""
        print("🔄 Real-time data scheduler started")

        # Track last update times
        last_weather_update = datetime.now() - timedelta(minutes=self.weather_update_interval)
        last_traffic_update = datetime.now() - timedelta(minutes=self.traffic_update_interval)
        last_construction_update = datetime.now() - timedelta(minutes=self.construction_update_interval)

        while self.running:
            try:
                now = datetime.now()

                # Check if weather update is due
                if (now - last_weather_update).total_seconds() >= self.weather_update_interval * 60:
                    self._update_weather_data()
                    last_weather_update = now

                # Check if traffic update is due
                if (now - last_traffic_update).total_seconds() >= self.traffic_update_interval * 60:
                    self._update_traffic_data()
                    last_traffic_update = now

                # Check if construction update is due
                if (now - last_construction_update).total_seconds() >= self.construction_update_interval * 60:
                    self._update_construction_data()
                    last_construction_update = now

                # Sleep for 30 seconds before next check
                time.sleep(30)

            except Exception as e:
                print(f"❌ Scheduler error: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(60)  # Wait longer on error

        print("⏸️ Real-time data scheduler stopped")

# Global data manager instance
real_data_manager = None

def get_real_data_manager():
    """Get real data manager instance"""
    return real_data_manager

def init_real_data_manager(socketio, event_manager, db_session_factory):
    """Initialize real data manager"""
    global real_data_manager
    real_data_manager = RealDataManager(socketio, event_manager, db_session_factory)
    return real_data_manager