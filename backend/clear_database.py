"""
Database Reset Script
Deletes old database with demo data to start fresh with real API data
"""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.database import SessionLocal, Hazard, ParkingSpot

def clear_all_data():
    """Clear all demo/old data from database"""
    db = SessionLocal()

    try:
        # Count existing data
        hazard_count = db.query(Hazard).count()
        parking_count = db.query(ParkingSpot).count()

        print(f"📊 Found in database:")
        print(f"   - {hazard_count} hazards")
        print(f"   - {parking_count} parking spots")

        if hazard_count > 0 or parking_count > 0:
            print("\n🗑️  Deleting all old data...")

            # Delete all hazards
            if hazard_count > 0:
                db.query(Hazard).delete()
                print(f"   ✅ Deleted {hazard_count} hazards")

            # Delete all parking spots
            if parking_count > 0:
                db.query(ParkingSpot).delete()
                print(f"   ✅ Deleted {parking_count} parking spots")

            db.commit()
            print("\n✅ Database cleared successfully!")
            print("ℹ️  Restart your backend to load fresh data from APIs")
        else:
            print("\n✅ Database is already empty")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == '__main__':
    print("🔄 Database Reset Script")
    print("=" * 50)

    response = input("\nThis will DELETE ALL DATA from the database. Continue? (yes/no): ")

    if response.lower() == 'yes':
        clear_all_data()
    else:
        print("❌ Cancelled")

