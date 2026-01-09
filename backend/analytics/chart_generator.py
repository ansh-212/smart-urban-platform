"""
Chart Generator using Plotly
Creates interactive charts for the analytics dashboard
"""
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import json

class ChartGenerator:
    def __init__(self):
        self.colors = {
            'primary': '#667eea',
            'success': '#2ecc71',
            'danger': '#ff4757',
            'warning': '#ffa502',
            'info': '#3742fa'
        }

    def create_hazard_trend_chart(self, trend_data):
        """Create line chart of hazards over time (hourly or daily)"""
        if not trend_data['daily_trends']:
            return self._empty_chart("No hazard data available")

        # Use the actual time labels (hours or dates)
        time_labels = [item['hour'] if 'hour' in item else item['date'] for item in trend_data['daily_trends']]
        counts = [item['count'] for item in trend_data['daily_trends']]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_labels,
            y=counts,
            mode='lines+markers',
            name='Hazard Reports',
            line=dict(color=self.colors['danger'], width=3),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(255, 71, 87, 0.1)'
        ))

        # Dynamic title based on trend type
        title = trend_data.get('trend_label', 'Hazard Reports Trend')
        x_title = 'Time (Hour)' if 'hourly' in title.lower() else 'Date'

        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title='Number of Hazards',
            template='plotly_white',
            height=300,
            showlegend=False
        )

        return fig.to_json()

    def create_severity_pie_chart(self, trend_data):
        """Create pie chart of hazard severity distribution"""
        if not trend_data['severity_distribution']:
            return self._empty_chart("No severity data")

        severities = [item['severity'] for item in trend_data['severity_distribution']]
        counts = [item['count'] for item in trend_data['severity_distribution']]

        colors = {
            'high': self.colors['danger'],
            'medium': self.colors['warning'],
            'low': self.colors['success']
        }

        chart_colors = [colors.get(sev, self.colors['primary']) for sev in severities]

        fig = go.Figure(data=[go.Pie(
            labels=severities,
            values=counts,
            marker_colors=chart_colors,
            hole=0.4
        )])

        fig.update_layout(
            title='Hazard Severity Distribution',
            template='plotly_white',
            height=300
        )

        return fig.to_json()

    def create_hourly_activity_chart(self, trend_data):
        """Create bar chart of hourly activity"""
        if not trend_data['hourly_distribution']:
            # Create sample data for 24 hours
            hours = list(range(24))
            counts = [0] * 24
        else:
            hours = [item['hour'] for item in trend_data['hourly_distribution']]
            counts = [item['count'] for item in trend_data['hourly_distribution']]

        fig = go.Figure(data=[go.Bar(
            x=hours,
            y=counts,
            marker_color=self.colors['info'],
            name='Hourly Reports'
        )])

        fig.update_layout(
            title='Hourly Activity Pattern',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Reports',
            template='plotly_white',
            height=300
        )

        return fig.to_json()

    def create_parking_utilization_chart(self, parking_data):
        """Create donut chart for parking utilization"""
        available = parking_data['available_spots']
        occupied = parking_data['occupied_spots']

        if available + occupied == 0:
            return self._empty_chart("No parking data")

        fig = go.Figure(data=[go.Pie(
            labels=['Available', 'Occupied'],
            values=[available, occupied],
            marker_colors=[self.colors['success'], self.colors['danger']],
            hole=0.6
        )])

        fig.update_layout(
            title='Parking Utilization',
            template='plotly_white',
            height=300,
            annotations=[dict(text=f"{parking_data['utilization_rate']}%",
                              x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        return fig.to_json()

    def create_response_time_chart(self, days=7):
        """Create response time trend chart (simulated data)"""
        # Simulate response time data for demo
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        times = [round(2.5 + i * 0.1 + (i % 3) * 0.3, 1) for i in range(days)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=times,
            mode='lines+markers',
            name='Response Time',
            line=dict(color=self.colors['warning'], width=3),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title='Average Response Time',
            xaxis_title='Date',
            yaxis_title='Minutes',
            template='plotly_white',
            height=300
        )

        return fig.to_json()

    def _empty_chart(self, message="No data available"):
        """Create empty chart with message"""
        fig = go.Figure()

        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray")
        )

        fig.update_layout(
            template='plotly_white',
            height=300,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )

        return fig.to_json()

def get_chart_generator():
    """Factory function to get chart generator"""
    return ChartGenerator()