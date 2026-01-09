"""
Real-Time Hazard Service
Loads hazards from real API sources:
- Weather API (floods, storms, ice)
- Construction API (road closures)
- Traffic API (accidents, congestion)

Auto-refreshes every sync to prevent accumulation.
"""
from datetime import datetime, timedelta
from typing import List, Dict
from models.database import SessionLocal, Hazard


class RealHazardService:
    def __init__(self, weather_service, construction_service, traffic_service):
        self.weather_service = weather_service
        self.construction_service = construction_service
        self.traffic_service = traffic_service
        print("🚨 Real Hazard Service initialized")

    def clear_api_hazards(self):
        """Clear old API-sourced hazards (not user-uploaded AI hazards)"""
        db = SessionLocal()
        try:
            # API hazard types (not from user uploads)
            api_hazard_types = ['flood', 'storm', 'ice', 'construction', 'road_closure', 'accident', 'congestion']

            # Delete old API hazards
            deleted = db.query(Hazard).filter(
                Hazard.type.in_(api_hazard_types)
            ).delete(synchronize_session='fetch')

            db.commit()

            if deleted > 0:
                print(f"🗑️  Cleared {deleted} old API hazards")

            return deleted

        except Exception as e:
            print(f"⚠️ Error clearing old hazards: {e}")
            db.rollback()
            return 0
        finally:
            db.close()

    def sync_weather_hazards(self):
        """Sync hazards from weather API"""
        try:
            weather_hazards = self.weather_service.get_weather_hazards()
            db = SessionLocal()

            hazards_created = 0
            for hazard_data in weather_hazards:
                hazard = Hazard(
                    type=hazard_data['type'],
                    latitude=hazard_data['location']['lat'],
                    longitude=hazard_data['location']['lon'],
                    severity=hazard_data['severity'],
                    description=hazard_data['description']
                )
                db.add(hazard)
                hazards_created += 1

            db.commit()
            db.close()

            if hazards_created > 0:
                print(f"🌤️ Synced {hazards_created} weather hazards")

            return hazards_created

        except Exception as e:
            print(f"⚠️ Error syncing weather hazards: {e}")
            return 0

    def sync_construction_hazards(self):
        """Sync hazards from construction/road closure API"""
        try:
            construction_zones = self.construction_service.get_construction_zones()
            road_closures = self.construction_service.get_road_closures()

            db = SessionLocal()
            hazards_created = 0

            # Add construction zones as hazards
            for zone in construction_zones:
                hazard = Hazard(
                    type='construction',
                    latitude=zone['location']['lat'],
                    longitude=zone['location']['lon'],
                    severity=zone.get('severity', 'medium'),
                    description=zone.get('description', 'Construction zone')
                )
                db.add(hazard)
                hazards_created += 1

            # Add road closures as hazards
            for closure in road_closures:
                hazard = Hazard(
                    type='road_closure',
                    latitude=closure['location']['lat'],
                    longitude=closure['location']['lon'],
                    severity=closure.get('severity', 'high'),
                    description=closure.get('reason', 'Road closure')
                )
                db.add(hazard)
                hazards_created += 1

            db.commit()
            db.close()

            if hazards_created > 0:
                print(f"🚧 Synced {hazards_created} construction/closure hazards")

            return hazards_created

        except Exception as e:
            print(f"⚠️ Error syncing construction hazards: {e}")
            return 0

    def sync_traffic_hazards(self):
        """Sync hazards from traffic incidents API"""
        try:
            incidents = self.traffic_service.get_traffic_incidents()

            db = SessionLocal()
            hazards_created = 0

            for incident in incidents:
                hazard = Hazard(
                    type=incident['type'],
                    latitude=incident['location']['lat'],
                    longitude=incident['location']['lon'],
                    severity=incident.get('severity', 'high'),
                    description=incident.get('description', 'Traffic incident')
                )
                db.add(hazard)
                hazards_created += 1

            db.commit()
            db.close()

            if hazards_created > 0:
                print(f"🚗 Synced {hazards_created} traffic incident hazards")

            return hazards_created

        except Exception as e:
            print(f"⚠️ Error syncing traffic hazards: {e}")
            return 0

    def sync_all_hazards(self):
        """Sync hazards from all real sources (clears old API hazards first)"""
        print("🔄 Syncing hazards from real APIs...")

        # IMPORTANT: Clear old API hazards first to prevent accumulation
        cleared = self.clear_api_hazards()

        # Now sync fresh data from APIs
        weather_count = self.sync_weather_hazards()
        construction_count = self.sync_construction_hazards()
        traffic_count = self.sync_traffic_hazards()

        total = weather_count + construction_count + traffic_count

        if total > 0:
            print(f"✅ Synced {total} total hazards from real sources")
        else:
            print("ℹ️  No new hazards from APIs")

        return {
            'cleared': cleared,
            'weather': weather_count,
            'construction': construction_count,
            'traffic': traffic_count,
            'total': total
        }


# Singleton instance
_real_hazard_service = None

def init_real_hazard_service(weather_service, construction_service, traffic_service):
    """Initialize the real hazard service"""
    global _real_hazard_service
    if _real_hazard_service is None:
        _real_hazard_service = RealHazardService(
            weather_service,
            construction_service,
            traffic_service
        )
    return _real_hazard_service

def get_real_hazard_service():
    """Get the real hazard service instance"""
    return _real_hazard_service

