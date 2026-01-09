"""
WebSocket Event Manager
Handles real-time events and broadcasting
"""
from flask_socketio import emit
import random
from datetime import datetime
from typing import Dict, List

class WebSocketEvents:
    def __init__(self, socketio):
        self.socketio = socketio
        self.connected_users = {}
        self.active_vehicles = {}

    def user_connected(self, sid: str, user_data: Dict = None):
        """Handle user connection"""
        self.connected_users[sid] = {
            'connected_at': datetime.now(),
            'user_data': user_data or {}
        }

        # Emit current system status to new user
        self.emit_system_status(to_user=sid)

        print(f"👤 User connected: {sid} (Total: {len(self.connected_users)})")

    def user_disconnected(self, sid: str):
        """Handle user disconnection"""
        if sid in self.connected_users:
            del self.connected_users[sid]
        print(f"👤 User disconnected: {sid} (Total: {len(self.connected_users)})")

    def broadcast_hazard_created(self, hazard_data: Dict):
        """Broadcast new hazard to all users"""
        self.socketio.emit('hazard_created', {
            'type': 'hazard_created',
            'data': hazard_data,
            'timestamp': datetime.now().isoformat(),
            'message': f"🚨 New {hazard_data['severity']} hazard detected: {hazard_data['type']}"
        })
        print(f"📡 Broadcasting hazard: {hazard_data['type']}")

    def broadcast_hazard_deleted(self, hazard_id: int):
        """Broadcast hazard deletion"""
        self.socketio.emit('hazard_deleted', {
            'type': 'hazard_deleted',
            'hazard_id': hazard_id,
            'timestamp': datetime.now().isoformat(),
            'message': "✅ Hazard resolved"
        })

    def broadcast_parking_updated(self, parking_data: Dict):
        """Broadcast parking status change"""
        self.socketio.emit('parking_updated', {
            'type': 'parking_updated',
            'data': parking_data,
            'timestamp': datetime.now().isoformat()
        })

    def broadcast_vehicle_update(self, vehicle_data: Dict):
        """Broadcast vehicle location update"""
        self.socketio.emit('vehicle_update', {
            'type': 'vehicle_update',
            'data': vehicle_data,
            'timestamp': datetime.now().isoformat()
        })

    def broadcast_stats_update(self, stats_data: Dict):
        """Broadcast updated system statistics"""
        self.socketio.emit('stats_update', {
            'type': 'stats_update',
            'data': stats_data,
            'timestamp': datetime.now().isoformat()
        })

    def emit_system_status(self, to_user: str = None):
        """Send current system status to user(s)"""
        status = {
            'type': 'system_status',
            'connected_users': len(self.connected_users),
            'active_vehicles': 0,  # Will be updated by vehicle simulator
            'timestamp': datetime.now().isoformat(),
            'message': f"🌐 Connected to real-time monitoring"
        }

        if to_user:
            self.socketio.emit('system_status', status, room=to_user)
        else:
            self.socketio.emit('system_status', status)

    def broadcast_emergency_alert(self, alert_data: Dict):
        """Broadcast emergency SOS alert"""
        try:
            print(f"📡 Broadcasting emergency alert to {len(self.connected_users)} users")

            broadcast_data = {
                'type': 'emergency_alert',
                'data': alert_data,
                'timestamp': datetime.now().isoformat(),
                'message': f"🆘 EMERGENCY ALERT: {alert_data.get('message', 'SOS activated')}"
            }

            print(f"📦 Broadcast data prepared")

            # Correct Flask-SocketIO syntax for broadcasting to all clients
            self.socketio.emit('emergency_alert', broadcast_data)
            print(f"✅ Emergency alert broadcasted successfully")

        except Exception as e:
            print(f"❌ Error in broadcast_emergency_alert: {e}")
            import traceback
            print(f"❌ Broadcast traceback: {traceback.format_exc()}")
            raise e



    def emit_system_status(self, to_user: str = None):
        """Send current system status to user(s)"""
        status = {
            'type': 'system_status',
            'connected_users': len(self.connected_users),
            'active_vehicles': len(self.active_vehicles),
            'timestamp': datetime.now().isoformat(),
            'message': f"🌐 Connected to real-time monitoring"
        }

        if to_user:
            self.socketio.emit('system_status', status, room=to_user)
        else:
            self.socketio.emit('system_status', status, broadcast=True)

    def get_connection_count(self) -> int:
        """Get number of connected users"""
        return len(self.connected_users)

# Global event manager (initialized in app.py)
event_manager = None

def get_event_manager():
    return event_manager

def init_event_manager(socketio):
    global event_manager
    event_manager = WebSocketEvents(socketio)
    return event_manager