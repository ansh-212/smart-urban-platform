from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class Hazard(Base):
    __tablename__ = 'hazards'

    id = Column(Integer, primary_key=True)
    type = Column(String(50))  # pothole, flood, accident, construction
    latitude = Column(Float)
    longitude = Column(Float)
    severity = Column(String(20))  # low, medium, high
    description = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'location': {
                'lat': self.latitude,
                'lon': self.longitude
            },
            'severity': self.severity,
            'description': self.description,
            'timestamp': self.timestamp.isoformat()
        }

class ParkingSpot(Base):
    __tablename__ = 'parking_spots'

    id = Column(Integer, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(String(20))  # available, occupied
    type = Column(String(50))  # regular, ev_charging, disabled
    price_per_hour = Column(Float)

    def to_dict(self):
        return {
            'id': self.id,
            'location': {
                'lat': self.latitude,
                'lon': self.longitude
            },
            'status': self.status,
            'type': self.type,
            'price_per_hour': self.price_per_hour
        }

# Database setup
engine = create_engine('sqlite:///urban_monitoring.db', echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        pass

def init_demo_data():
    """
    Initialize database tables (no demo data needed anymore)

    All data now comes from real sources:
    - Hazards: Weather API, Construction API, Traffic API, AI Detection
    - Parking: OpenStreetMap (OSM) API (free, real-time)
    - Gas Stations: OpenStreetMap (OSM) API (free, real-time)
    - EV Charging: OpenStreetMap (OSM) API (free, real-time)
    """
    print("✅ Database initialized")
    print("ℹ️  All data loaded from real APIs:")
    print("   🚨 Hazards: Weather, Construction, Traffic APIs + AI Detection")
    print("   🅿️ Parking: OpenStreetMap (OSM)")
    print("   ⛽ Gas Stations: OpenStreetMap (OSM)")
    print("   🔌 EV Charging: OpenStreetMap (OSM)")

