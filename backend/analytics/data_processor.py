"""
Data Analytics Processor
Processes database data for charts and insights
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

class DataProcessor:
    def __init__(self):
        pass  # No database session needed - we'll pass data directly

    def get_hazard_trends(self, hazards_list: List[Dict], days: int = 7) -> Dict[str, Any]:
        """Get hazard trends from hazards list"""
        df = pd.DataFrame(hazards_list)

        if df.empty:
            return self._empty_trend_data()

        # For real-time demo, use HOURLY trends for today
        df['datetime'] = pd.to_datetime(df['timestamp'])
        today = pd.Timestamp.now().date()

        # Filter to today's data for hourly trending
        today_data = df[df['datetime'].dt.date == today].copy()

        if not today_data.empty:
            # Group by hour for today (more responsive to real-time changes)
            today_data['hour'] = today_data['datetime'].dt.strftime('%H:00')
            hourly_counts = today_data.groupby('hour').size().reset_index(name='count')

            # Fill missing hours with 0 for complete 24-hour view
            all_hours = [f"{i:02d}:00" for i in range(24)]
            hourly_complete = pd.DataFrame({'hour': all_hours})
            hourly_counts = hourly_complete.merge(hourly_counts, on='hour', how='left').fillna(0)
            hourly_counts['count'] = hourly_counts['count'].astype(int)

            trend_data = hourly_counts.to_dict('records')
            trend_label = "Today's Hourly Trends"
        else:
            # Fallback to daily trends if no today data
            df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
            daily_counts = df.groupby('date').size().reset_index(name='count')
            trend_data = daily_counts.to_dict('records')
            trend_label = "Daily Trends"

        # Group by severity
        severity_counts = df.groupby('severity').size().reset_index(name='count')

        # Group by type
        type_counts = df.groupby('type').size().reset_index(name='count')

        # Overall hourly distribution (all time)
        df['hour'] = df['datetime'].dt.hour
        hourly_dist = df.groupby('hour').size().reset_index(name='count')

        return {
            'daily_trends': trend_data,
            'trend_label': trend_label,
            'severity_distribution': severity_counts.to_dict('records'),
            'type_distribution': type_counts.to_dict('records'),
            'hourly_distribution': hourly_dist.to_dict('records'),
            'total_hazards': len(hazards_list),
            'high_severity_count': len([h for h in hazards_list if h.get('severity') == 'high']),
            'last_updated': datetime.now().isoformat()
        }

    def get_parking_analytics(self, parking_list: List[Dict]) -> Dict[str, Any]:
        """Get parking utilization analytics from parking list"""
        if not parking_list:
            return {
                'total_spots': 0,
                'available_spots': 0,
                'occupied_spots': 0,
                'utilization_rate': 0,
                'type_distribution': [],
                'availability_by_type': [],
                'average_price': 0
            }

        total = len(parking_list)
        available = len([s for s in parking_list if s.get('status') == 'available'])
        utilization = ((total - available) / total * 100) if total > 0 else 0

        # Convert to DataFrame for analysis
        df = pd.DataFrame(parking_list)

        if not df.empty:
            type_dist = df.groupby('type').size().reset_index(name='count')
            avg_price = df['price_per_hour'].mean() if 'price_per_hour' in df.columns else 0

            # Availability by type
            if 'status' in df.columns:
                avail_by_type = df.groupby(['type', 'status']).size().unstack(fill_value=0).reset_index()
                avail_data = avail_by_type.to_dict('records') if not avail_by_type.empty else []
            else:
                avail_data = []
        else:
            type_dist = pd.DataFrame()
            avg_price = 0
            avail_data = []

        return {
            'total_spots': total,
            'available_spots': available,
            'occupied_spots': total - available,
            'utilization_rate': round(utilization, 1),
            'type_distribution': type_dist.to_dict('records') if not type_dist.empty else [],
            'availability_by_type': avail_data,
            'average_price': round(avg_price, 2)
        }

    def get_real_time_metrics(self, hazards_list: List[Dict], parking_list: List[Dict]) -> Dict[str, Any]:
        """Get current real-time metrics"""
        total_hazards = len(hazards_list)
        high_severity = len([h for h in hazards_list if h.get('severity') == 'high'])
        total_parking = len(parking_list)
        available_parking = len([p for p in parking_list if p.get('status') == 'available'])

        # Calculate trends (simulate for demo)
        hazard_trend = random.choice(['+5%', '+12%', '-3%', '+8%'])
        parking_trend = random.choice(['+2%', '-7%', '+15%', '-1%'])

        # Average response time (simulated)
        avg_response_time = round(random.uniform(1.5, 4.5), 1)
        response_trend = random.choice(['-0.3min', '+0.1min', '-0.8min'])

        return {
            'total_hazards': {
                'value': total_hazards,
                'trend': hazard_trend,
                'label': 'Today\'s Hazards'
            },
            'high_severity_hazards': {
                'value': high_severity,
                'trend': '+2',
                'label': 'High Priority'
            },
            'parking_utilization': {
                'value': f"{round((total_parking - available_parking) / total_parking * 100 if total_parking > 0 else 0, 1)}%",
                'trend': parking_trend,
                'label': 'Parking Used'
            },
            'avg_response_time': {
                'value': f"{avg_response_time}min",
                'trend': response_trend,
                'label': 'Avg Response'
            },
            'connected_users': {
                'value': random.randint(15, 45),
                'trend': '+3',
                'label': 'Active Users'
            }
        }

    def get_heat_map_data(self, hazards_list: List[Dict]) -> Dict[str, Any]:
        """Get data for heat map visualization"""
        heat_points = []
        for h in hazards_list:
            # Weight by severity
            severity = h.get('severity', 'medium')
            weight = {'low': 0.3, 'medium': 0.6, 'high': 1.0}.get(severity, 0.5)

            location = h.get('location', {})
            if 'lat' in location and 'lon' in location:
                heat_points.append({
                    'lat': location['lat'],
                    'lng': location['lon'],  # lng for heat map libraries
                    'weight': weight,
                    'type': h.get('type', 'unknown'),
                    'severity': severity
                })

        return {
            'heat_points': heat_points,
            'total_points': len(heat_points)
        }

    def _empty_trend_data(self):
        """Return empty data structure for trends"""
        return {
            'daily_trends': [],
            'trend_label': 'No Data Available',
            'severity_distribution': [],
            'type_distribution': [],
            'hourly_distribution': [],
            'total_hazards': 0,
            'high_severity_count': 0,
            'last_updated': datetime.now().isoformat()
        }

def get_data_processor():
    """Factory function to get data processor"""
    return DataProcessor()