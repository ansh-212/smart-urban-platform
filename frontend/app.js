// At the top of app.js, replace hardcoded URLs:

// Detect if running locally or in production
const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = IS_LOCAL ? 'http://localhost:5000/api' : '/api';
const SOCKET_URL = IS_LOCAL ? 'http://localhost:5000' : '';

// Initialize Socket.IO connection
const socket = io(SOCKET_URL);

// Connection status
let isConnected = false;
let connectionDot = document.getElementById('connectionDot');
let connectionText = document.getElementById('connectionText');
let liveIndicator = document.getElementById('liveIndicator');

// Socket event listeners
socket.on('connect', function() {
    isConnected = true;
    updateConnectionStatus('Connected', 'connected');
    liveIndicator.classList.add('active');
    showNotification('🌐 Connected to real-time monitoring', 'success');

    // Request current data on connect
    socket.emit('request_current_data');
});

socket.on('disconnect', function() {
    isConnected = false;
    updateConnectionStatus('Disconnected', 'disconnected');
    liveIndicator.classList.remove('active');
    showNotification('❌ Lost connection to server', 'error');
});

socket.on('connected', function(data) {
    console.log('Connected to server:', data.message);
});

// Real-time event handlers
socket.on('hazard_created', function(data) {
    console.log('New hazard:', data);
    addHazardToMap(data.data);
    updateStats();
    showNotification(data.message, 'warning');
    playNotificationSound();
});

socket.on('hazard_deleted', function(data) {
    console.log('Hazard deleted:', data);
    removeHazardFromMap(data.hazard_id);
    updateStats();
    showNotification(data.message, 'success');
});

socket.on('parking_updated', function(data) {
    console.log('Parking updated:', data);
    updateParkingOnMap(data.data);
    updateStats();
});

socket.on('emergency_alert', function(data) {
    console.log('Emergency alert:', data);
    showEmergencyAlert(data);
    playEmergencySound();
});

socket.on('stats_update', function(data) {
    console.log('Stats update:', data);
    updateStatsDisplay(data.data);
});

socket.on('current_data', function(data) {
    console.log('Received current data');

    // Load all current data
    loadHazardsFromData(data.hazards);
    loadParkingFromData(data.parking);

    updateStats();
});

socket.on('hazards_refreshed', function(data) {
    console.log('Hazards refreshed from APIs:', data);

    // Reload hazards from server
    fetch(`${API_BASE}/hazards`)
        .then(response => response.json())
        .then(hazards => {
            loadHazardsFromData(hazards);
            updateStats();
            showNotification(
                `🔄 Hazards refreshed: ${data.cleared} old removed, ${data.total} new loaded`,
                'info',
                5000
            );
        })
        .catch(error => {
            console.error('Error reloading hazards:', error);
        });
});

// ==================== MAP INITIALIZATION ====================

// Initialize map
const map = L.map('map').setView([37.7749, -122.4194], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Layer groups
const layers = {
    hazards: L.layerGroup().addTo(map),
    parking: L.layerGroup(),
    crowd: L.layerGroup(),
    route: L.layerGroup(),
    gasStations: L.layerGroup(),
    evCharging: L.layerGroup()
};

// ==================== ICONS AND MARKERS ====================

const icons = {
    hazard_high: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #ff4757; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">⚠️</div>'
    }),
    hazard_medium: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #ffa502; width: 25px; height: 25px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">⚠️</div>'
    }),
    parking_available: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #2ecc71; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">P</div>'
    }),
    parking_occupied: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #e74c3c; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">P</div>'
    }),
    parking_osm: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #3498db; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; border: 2px solid white;">P</div>'
    }),
    gas_station: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #e74c3c; width: 25px; height: 25px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">⛽</div>'
    }),
    ev_charging: L.divIcon({
        className: 'custom-icon',
        html: '<div style="background: #27ae60; width: 25px; height: 25px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">🔌</div>'
    })
};

// Store markers by ID for updates
const hazardMarkers = new Map();
const parkingMarkers = new Map();

// ==================== UTILITY FUNCTIONS ====================

function updateConnectionStatus(text, status) {
    connectionText.textContent = text;
    connectionDot.className = `status-indicator ${status}`;
}

function showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notifications');

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
        <div style="font-weight: bold; margin-bottom: 5px;">${new Date().toLocaleTimeString()}</div>
        <div>${message}</div>
    `;

    container.appendChild(notification);

    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, duration);
}

function showEmergencyAlert(data) {
    showNotification(
        `🆘 EMERGENCY ALERT: ${data.data.message}`,
        'emergency',
        10000
    );

    // Flash the page briefly
    document.body.style.background = '#ff4757';
    setTimeout(() => {
        document.body.style.background = '#f5f5f5';
    }, 500);
}

function playNotificationSound() {
    // Simple beep sound
    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmE');
    audio.volume = 0.3;
    audio.play().catch(e => console.log('Audio play failed:', e));
}

function playEmergencySound() {
    // Emergency sound pattern
    for(let i = 0; i < 3; i++) {
        setTimeout(() => {
            playNotificationSound();
        }, i * 200);
    }
}

// ==================== MAP DATA FUNCTIONS ====================

function addHazardToMap(hazard) {
    const icon = hazard.severity === 'high' ? icons.hazard_high : icons.hazard_medium;

    const marker = L.marker(
        [hazard.location.lat, hazard.location.lon],
        { icon }
    ).addTo(layers.hazards);

    marker.bindPopup(`
        <strong>${hazard.type.toUpperCase()}</strong><br>
        Severity: ${hazard.severity}<br>
        ${hazard.description}<br>
        <small>${new Date(hazard.timestamp).toLocaleString()}</small>
        <br><br>
        <button onclick="deleteHazard(${hazard.id})" style="background: #ff4757; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">
            Delete
        </button>
    `);

    hazardMarkers.set(hazard.id, marker);
}

function removeHazardFromMap(hazardId) {
    if (hazardMarkers.has(hazardId)) {
        layers.hazards.removeLayer(hazardMarkers.get(hazardId));
        hazardMarkers.delete(hazardId);
    }
}

function updateParkingOnMap(parking) {
    // Remove existing marker
    if (parkingMarkers.has(parking.id)) {
        layers.parking.removeLayer(parkingMarkers.get(parking.id));
    }

    // Add updated marker
    const icon = parking.status === 'available' ? icons.parking_available : icons.parking_occupied;

    const marker = L.marker(
        [parking.location.lat, parking.location.lon],
        { icon }
    ).addTo(layers.parking);

    marker.bindPopup(`
        <strong>Parking Spot #${parking.id}</strong><br>
        Status: ${parking.status}<br>
        Type: ${parking.type}<br>
        Rate: $${parking.price_per_hour}/hr
        <br><br>
        <button onclick="toggleParkingStatus(${parking.id})" style="background: #2ecc71; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">
            Toggle Status
        </button>
    `);

    parkingMarkers.set(parking.id, marker);
}


// ==================== DATA LOADING FUNCTIONS ====================

function loadHazardsFromData(hazards) {
    layers.hazards.clearLayers();
    hazardMarkers.clear();

    hazards.forEach(hazard => {
        addHazardToMap(hazard);
    });

    document.getElementById('hazardCount').textContent = hazards.length;
}

function loadParkingFromData(spots) {
    layers.parking.clearLayers();
    parkingMarkers.clear();

    const available = spots.filter(s => s.status === 'available').length;

    spots.forEach(spot => {
        updateParkingOnMap(spot);
    });

    document.getElementById('parkingCount').textContent = available;
}

// ==================== API INTERACTION FUNCTIONS ====================

async function deleteHazard(hazardId) {
    try {
        const response = await fetch(`${API_BASE}/hazards/${hazardId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // Real-time update will be handled by WebSocket
            showNotification('✅ Hazard deletion requested', 'success');
        } else {
            showNotification('❌ Failed to delete hazard', 'error');
        }
    } catch (error) {
        console.error('Error deleting hazard:', error);
        showNotification('❌ Error deleting hazard', 'error');
    }
}

async function toggleParkingStatus(spotId) {
    try {
        const response = await fetch(`${API_BASE}/parking/${spotId}/toggle`, {
            method: 'POST'
        });

        if (response.ok) {
            showNotification('🅿️ Parking status updated', 'success');
        } else {
            showNotification('❌ Failed to update parking', 'error');
        }
    } catch (error) {
        console.error('Error toggling parking:', error);
        showNotification('❌ Error updating parking', 'error');
    }
}

// ==================== STATS UPDATE ====================

function updateStatsDisplay(stats) {
    if (stats.total_hazards !== undefined) {
        document.getElementById('hazardCount').textContent = stats.total_hazards;
    }
    if (stats.available_parking !== undefined) {
        document.getElementById('parkingCount').textContent = stats.available_parking;
    }
    if (stats.connected_users !== undefined) {
        document.getElementById('userCount').textContent = stats.connected_users;
    }
}

async function updateStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const stats = await response.json();
        updateStatsDisplay(stats);
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// ==================== EXISTING FUNCTIONS (Updated) ====================

// Toggle layers
function toggleLayer(layerName) {
    if (map.hasLayer(layers[layerName])) {
        map.removeLayer(layers[layerName]);
    } else {
        // For parking, gas stations, and EV charging - prompt user to click location
        if (layerName === 'parking') {
            enableLocationSelection('parking');
            return;
        }
        if (layerName === 'gasStations') {
            enableLocationSelection('gas');
            return;
        }
        if (layerName === 'evCharging') {
            enableLocationSelection('ev');
            return;
        }

        map.addLayer(layers[layerName]);

        // Request fresh data if needed
        if (layerName === 'crowd') loadCrowdDensity();
    }
}

// Location selection state for amenities
let amenityLocationState = {
    active: false,
    type: null,
    marker: null,
    lat: null,  // Store search location coordinates
    lon: null   // for route calculation
};

// Custom location marker icon
const locationSelectIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

function enableLocationSelection(type) {
    const messages = {
        'parking': '📍 Click on map to find parking spots nearby',
        'gas': '📍 Click on map to find gas stations nearby',
        'ev': '📍 Click on map to find EV charging stations nearby'
    };

    showNotification(messages[type], 'info', 8000);

    amenityLocationState.active = true;
    amenityLocationState.type = type;

    // Change cursor to crosshair
    map.getContainer().style.cursor = 'crosshair';

    // Add click handler
    map.once('click', function(e) {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;

        // IMPORTANT: Store the search location coordinates
        amenityLocationState.lat = lat;
        amenityLocationState.lon = lon;

        // Remove old marker if exists
        if (amenityLocationState.marker) {
            amenityLocationState.marker.remove();
        }

        // Add marker at selected location
        amenityLocationState.marker = L.marker([lat, lon], {
            icon: locationSelectIcon
        }).addTo(map);

        amenityLocationState.marker.bindPopup('<b>📍 Search Location</b><br>Loading nearby amenities...').openPopup();

        // Reset cursor
        map.getContainer().style.cursor = '';

        // Load data based on type
        if (type === 'parking') {
            loadOSMParkingSpots(lat, lon);
        } else if (type === 'gas') {
            loadOSMGasStations(lat, lon);
        } else if (type === 'ev') {
            loadOSMEVChargingStations(lat, lon);
        }

        amenityLocationState.active = false;
    });
}

// Load crowd density (unchanged)
async function loadCrowdDensity() {
    try {
        const response = await fetch(`${API_BASE}/crowd-density`);
        const zones = await response.json();

        layers.crowd.clearLayers();

        const highDensity = zones.filter(z => z.density === 'high').length;

        zones.forEach(zone => {
            const color = zone.density === 'high' ? '#ff4757' :
                zone.density === 'medium' ? '#ffa502' : '#2ecc71';

            const circle = L.circle(
                [zone.center.lat, zone.center.lon],
                {
                    radius: zone.radius,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.3
                }
            ).addTo(layers.crowd);

            circle.bindPopup(`
                <strong>${zone.name}</strong><br>
                Density: ${zone.density}<br>
                Est. Count: ${zone.estimated_count}
            `);
        });

        document.getElementById('crowdCount').textContent = highDensity;

    } catch (error) {
        console.error('Error loading crowd data:', error);
    }
}

// Calculate safe route - UPDATED FOR INTERACTIVE SELECTION
let routingState = {
    active: false,
    startMarker: null,
    endMarker: null,
    clickCount: 0
};

// Custom marker icons for routing
const startIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

const endIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

async function calculateRoute() {
    // Clear previous route
    layers.route.clearLayers();

    if (routingState.startMarker) {
        routingState.startMarker.remove();
    }
    if (routingState.endMarker) {
        routingState.endMarker.remove();
    }

    // Reset state
    routingState = {
        active: true,
        startMarker: null,
        endMarker: null,
        clickCount: 0
    };

    showNotification('🗺️ Click on map to set START point', 'info', 10000);

    // Enable map click handler
    map.on('click', onMapClickForRouting);
}

function onMapClickForRouting(e) {
    if (!routingState.active) return;

    if (routingState.clickCount === 0) {
        // Set start point
        routingState.startMarker = L.marker([e.latlng.lat, e.latlng.lng], {
            icon: startIcon,
            draggable: true
        }).addTo(map);

        routingState.startMarker.bindPopup('<b>START</b><br>Drag to reposition').openPopup();

        // Listen for drag events
        routingState.startMarker.on('dragend', function() {
            if (routingState.endMarker) {
                calculateRouteFromMarkers();
            }
        });

        routingState.clickCount = 1;
        showNotification('✅ Start set! Click to set END point', 'success', 10000);

    } else if (routingState.clickCount === 1) {
        // Set end point
        routingState.endMarker = L.marker([e.latlng.lat, e.latlng.lng], {
            icon: endIcon,
            draggable: true
        }).addTo(map);

        routingState.endMarker.bindPopup('<b>END</b><br>Drag to reposition').openPopup();

        // Listen for drag events
        routingState.endMarker.on('dragend', function() {
            calculateRouteFromMarkers();
        });

        routingState.clickCount = 2;
        routingState.active = false;

        // Remove map click handler
        map.off('click', onMapClickForRouting);

        // Calculate route
        calculateRouteFromMarkers();
    }
}

async function calculateRouteFromMarkers() {
    if (!routingState.startMarker || !routingState.endMarker) {
        return;
    }

    const start = {
        lat: routingState.startMarker.getLatLng().lat,
        lon: routingState.startMarker.getLatLng().lng
    };

    const end = {
        lat: routingState.endMarker.getLatLng().lat,
        lon: routingState.endMarker.getLatLng().lng
    };

    showNotification('🔄 Calculating route...', 'info');

    try {
        const response = await fetch(`${API_BASE}/route`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start, end })
        });

        const route = await response.json();

        // Clear previous route line
        layers.route.clearLayers();

        // Re-add markers
        routingState.startMarker.addTo(layers.route);
        routingState.endMarker.addTo(layers.route);

        // Draw route with ENHANCED HIGHLIGHTING
        // First, draw a thick white border for contrast
        const routeBorder = L.polyline(
            route.coordinates.map(c => [c[1], c[0]]),
            {
                color: '#ffffff',
                weight: 10,
                opacity: 0.8
            }
        ).addTo(layers.route);

        // Then draw the main route line with bright color
        const polyline = L.polyline(
            route.coordinates.map(c => [c[1], c[0]]),
            {
                color: '#3742fa',  // Bright blue for high visibility
                weight: 6,
                opacity: 1,
                dashArray: '10, 5',  // Animated dashed line
                className: 'route-animated'
            }
        ).addTo(layers.route);

        // Add distance/time labels along the route
        const midPoint = route.coordinates[Math.floor(route.coordinates.length / 2)];
        const routeLabel = L.marker([midPoint[1], midPoint[0]], {
            icon: L.divIcon({
                className: 'route-label',
                html: `<div style="background: rgba(55, 66, 250, 0.95); color: white; padding: 8px 12px; border-radius: 20px; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.3); white-space: nowrap; border: 2px solid white;">
                    📍 ${route.distance.toFixed(2)}km • ${route.duration.toFixed(0)} min
                </div>`
            })
        }).addTo(layers.route);

        // Add interactive popup to the route line
        polyline.bindPopup(`
            <div style="text-align: center;">
                <h3 style="margin: 5px 0; color: #3742fa;">🗺️ Route Details</h3>
                <div style="margin: 10px 0;">
                    <strong>Distance:</strong> ${route.distance.toFixed(2)} km<br>
                    <strong>ETA:</strong> ${route.duration.toFixed(0)} minutes<br>
                    <strong>Hazards Avoided:</strong> ${route.avoided_hazards || 0}
                </div>
                ${route.real_routing ?
                    '<span style="color: #2ecc71;">✓ Real-time routing</span>' :
                    '<span style="color: #f39c12;">⚠ Simulated routing</span>'}
            </div>
        `);

        // Fit map to route with nice padding
        const bounds = polyline.getBounds();
        map.fitBounds(bounds, { padding: [80, 80] });

        // IMPORTANT: Add the route layer to the map!
        if (!map.hasLayer(layers.route)) {
            map.addLayer(layers.route);
        }

        // Show route info
        const message = route.real_routing
            ? `🗺️ Route: ${route.distance.toFixed(2)}km, ETA: ${route.duration.toFixed(0)}min, Avoided ${route.avoided_hazards} hazards`
            : `🗺️ Route: ${route.distance.toFixed(2)}km, ETA: ${route.duration.toFixed(0)}min (Simulated)`;

        showNotification(message, 'success', 8000);

    } catch (error) {
        console.error('Error calculating route:', error);
        showNotification('❌ Error calculating route', 'error');
    }
}

// Calculate route TO an amenity (parking, gas station, EV charging)
async function calculateRouteToAmenity(destLat, destLon, amenityName, amenityType) {
    // Use the stored search location (blue marker) as start point
    // If no search location stored, fall back to map center
    const start = {
        lat: amenityLocationState.lat || map.getCenter().lat,
        lon: amenityLocationState.lon || map.getCenter().lng
    };

    const end = {
        lat: destLat,
        lon: destLon
    };

    showNotification(`🔄 Calculating route to ${amenityName}...`, 'info', 2000);

    try {
        const response = await fetch(`${API_BASE}/route`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ start, end })
        });

        const route = await response.json();

        // Clear previous route
        layers.route.clearLayers();

        // Add start marker (green) at the SEARCH LOCATION (blue marker position)
        const startMarker = L.marker([start.lat, start.lon], {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(layers.route);

        startMarker.bindPopup('<b>📍 Your Search Location</b><br>(Where you clicked)').openPopup();

        // Add destination marker (based on amenity type)
        let destIcon;
        if (amenityType === 'parking') {
            destIcon = icons.parking_osm;
        } else if (amenityType === 'gas') {
            destIcon = icons.gas_station;
        } else if (amenityType === 'ev') {
            destIcon = icons.ev_charging;
        }

        const endMarker = L.marker([destLon, destLat], { icon: destIcon })
            .addTo(layers.route);

        endMarker.bindPopup(`<b>🎯 ${amenityName}</b>`);

        // Draw route with highlighting (white border + blue line)
        const routeBorder = L.polyline(
            route.coordinates.map(c => [c[1], c[0]]),
            {
                color: '#ffffff',
                weight: 10,
                opacity: 0.8
            }
        ).addTo(layers.route);

        const polyline = L.polyline(
            route.coordinates.map(c => [c[1], c[0]]),
            {
                color: '#3742fa',
                weight: 6,
                opacity: 1,
                dashArray: '10, 5',
                className: 'route-animated'
            }
        ).addTo(layers.route);

        // Add route info label at midpoint
        const midPoint = route.coordinates[Math.floor(route.coordinates.length / 2)];
        const routeLabel = L.marker([midPoint[1], midPoint[0]], {
            icon: L.divIcon({
                className: 'route-label',
                html: `
                    <div style="background: white; padding: 8px 12px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-weight: bold; border: 2px solid #3742fa;">
                        📏 ${route.distance.toFixed(2)} km<br>
                        ⏱️ ${route.duration.toFixed(0)} min<br>
                        ${route.avoided_hazards > 0 ? `⚠️ Avoided ${route.avoided_hazards} hazards<br>` : ''}
                    </div>
                `
            })
        }).addTo(layers.route);

        // Fit map to route
        map.fitBounds(polyline.getBounds(), { padding: [80, 80] });

        // Show route layer
        if (!map.hasLayer(layers.route)) {
            map.addLayer(layers.route);
        }

        // Show notification
        showNotification(
            `✅ Route to ${amenityName}: ${route.distance.toFixed(2)}km, ${route.duration.toFixed(0)} minutes`,
            'success',
            8000
        );

    } catch (error) {
        console.error('Error calculating route to amenity:', error);
        showNotification('❌ Error calculating route', 'error');
    }
}

// Emergency SOS (updated with real emergency facilities from OSM)
async function sendSOS() {
    if (!confirm('Send emergency SOS alert?')) return;

    const location = map.getCenter();

    console.log('🆘 Sending SOS alert...');

    try {
        const response = await fetch(`${API_BASE}/sos`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                location: { lat: location.lat, lon: location.lng }
            })
        });

        console.log('SOS Response status:', response.status);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('SOS Response data:', result);

        // Clear previous emergency markers
        layers.route.clearLayers();

        // Show emergency facilities with custom icons and detailed popups
        if (result.nearest_safe_zones && result.nearest_safe_zones.length > 0) {
            result.nearest_safe_zones.forEach((zone, index) => {
                // Create custom icon based on facility type
                let iconHtml = '';
                let iconColor = '';

                switch(zone.type) {
                    case 'hospital':
                        iconHtml = '🏥';
                        iconColor = '#e74c3c';
                        break;
                    case 'police':
                        iconHtml = '👮';
                        iconColor = '#3498db';
                        break;
                    case 'fire_station':
                        iconHtml = '🚒';
                        iconColor = '#e67e22';
                        break;
                    default:
                        iconHtml = '🆘';
                        iconColor = '#95a5a6';
                }

                const customIcon = L.divIcon({
                    html: `<div style="
                        font-size: 24px;
                        text-align: center;
                        width: 40px;
                        height: 40px;
                        line-height: 40px;
                        background: ${iconColor};
                        border-radius: 50%;
                        border: 3px solid white;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                    ">${iconHtml}</div>`,
                    className: 'emergency-icon',
                    iconSize: [40, 40],
                    iconAnchor: [20, 20]
                });

                // Create detailed popup
                const popupContent = `
                    <div style="min-width: 200px;">
                        <h3 style="margin: 0 0 10px 0; color: ${iconColor}; font-size: 16px;">
                            ${iconHtml} ${zone.name}
                        </h3>
                        <div style="font-size: 12px; line-height: 1.6;">
                            <strong>Type:</strong> ${zone.type.replace('_', ' ').toUpperCase()}<br>
                            <strong>Distance:</strong> ${zone.distance.toFixed(2)} km<br>
                            ${zone.address !== 'N/A' ? `<strong>Address:</strong> ${zone.address}<br>` : ''}
                            ${zone.phone !== 'N/A' ? `<strong>Phone:</strong> <a href="tel:${zone.phone}">${zone.phone}</a><br>` : ''}
                        </div>
                        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
                            <button onclick="getDirections(${zone.location.lat}, ${zone.location.lon})" 
                                    style="background: ${iconColor}; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; width: 100%;">
                                📍 Get Directions
                            </button>
                        </div>
                    </div>
                `;

                const marker = L.marker([zone.location.lat, zone.location.lon], { icon: customIcon })
                    .addTo(layers.route)
                    .bindPopup(popupContent);

                // Open popup for the closest facility
                if (index === 0) {
                    marker.openPopup();
                }
            });

            map.addLayer(layers.route);

            // Show summary notification
            const facilityCount = result.facilities_count || {};
            const summary = `
                Found ${result.nearest_safe_zones.length} emergency facilities nearby:<br>
                🏥 Hospitals: ${facilityCount.hospitals || 0}<br>
                👮 Police: ${facilityCount.police || 0}<br>
                🚒 Fire Stations: ${facilityCount.fire_stations || 0}
            `;

            showNotification(`🆘 EMERGENCY ALERT SENT!<br>${summary}`, 'emergency', 15000);
        } else {
            showNotification('🆘 EMERGENCY ALERT SENT! No facilities found nearby.', 'emergency', 10000);
        }

    } catch (error) {
        console.error('SOS Error details:', error);
        showNotification(`❌ Emergency alert failed: ${error.message}`, 'error');
    }
}

// Helper function to get directions to emergency facility
function getDirections(lat, lon) {
    const userLocation = map.getCenter();
    // You can integrate with routing service here
    window.open(`https://www.google.com/maps/dir/${userLocation.lat},${userLocation.lng}/${lat},${lon}`, '_blank');
}

// Image upload functions (unchanged but with real-time notifications)
function previewImage() {
    const fileInput = document.getElementById('imageUpload');
    const preview = document.getElementById('imagePreview');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" style="width: 100%; border-radius: 5px;">`;
        };
        reader.readAsDataURL(file);
    }
}

async function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    const resultsDiv = document.getElementById('detectionResults');

    if (!file) {
        showNotification('❌ Please select an image first', 'error');
        return;
    }

    resultsDiv.innerHTML = '⏳ Analyzing with AI...';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch(`${API_BASE}/detect`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            let html = `<strong>✅ Detected ${result.total_count} objects:</strong><br>`;

            for (let [category, count] of Object.entries(result.counts_by_category)) {
                html += `${category}: ${count}<br>`;
            }

            html += '<br><small>';
            result.detections.slice(0, 5).forEach(d => {
                html += `• ${d.type} (${(d.confidence * 100).toFixed(0)}%)<br>`;
            });
            html += '</small>';

            resultsDiv.innerHTML = html;

            showNotification(`🤖 AI Detection Complete! Found ${result.total_count} objects`, 'success');
        } else {
            resultsDiv.innerHTML = '❌ Detection failed';
            showNotification('❌ Detection failed: ' + (result.error || 'Unknown error'), 'error');
        }

    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = '❌ Error';
        showNotification('❌ Error uploading image: ' + error.message, 'error');
    }
}

// Hazard location state
let hazardLocationState = {
    active: false,
    marker: null,
    selectedLocation: null
};

// Custom hazard location marker
const hazardLocationIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

async function uploadAndReport() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    const resultsDiv = document.getElementById('detectionResults');

    if (!file) {
        showNotification('❌ Please select an image first', 'error');
        return;
    }

    // Ask user to select location on map
    if (!hazardLocationState.selectedLocation) {
        showNotification('📍 Click on the map to set hazard location', 'info', 10000);
        resultsDiv.innerHTML = '📍 Click map to set location...';

        // Enable location selection mode
        hazardLocationState.active = true;

        // Add click handler
        map.once('click', function(e) {
            hazardLocationState.selectedLocation = {
                lat: e.latlng.lat,
                lon: e.latlng.lng
            };

            // Remove old marker if exists
            if (hazardLocationState.marker) {
                hazardLocationState.marker.remove();
            }

            // Add marker at selected location
            hazardLocationState.marker = L.marker([e.latlng.lat, e.latlng.lng], {
                icon: hazardLocationIcon
            }).addTo(map);

            hazardLocationState.marker.bindPopup('<b>Hazard Location</b><br>Click "Detect & Report" again').openPopup();

            resultsDiv.innerHTML = '✅ Location set! Click "Detect & Report" again.';
            showNotification('✅ Location set! Click "Detect & Report Hazards" again to confirm', 'success');

            hazardLocationState.active = false;
        });

        return;
    }

    // Now we have location, proceed with detection
    resultsDiv.innerHTML = '⏳ Detecting and reporting...';

    const formData = new FormData();
    formData.append('image', file);
    formData.append('location', JSON.stringify(hazardLocationState.selectedLocation));

    try {
        const response = await fetch(`${API_BASE}/detect-and-report`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            resultsDiv.innerHTML = `✅ Created ${result.hazards_created} hazard reports at selected location`;

            showNotification(
                `🚨 Hazards Reported! ${result.hazards_created} hazards created at the location you selected.`,
                'warning'
            );

            // Remove location marker
            if (hazardLocationState.marker) {
                hazardLocationState.marker.remove();
            }

            // Reset state
            hazardLocationState.selectedLocation = null;
            hazardLocationState.marker = null;

        } else {
            resultsDiv.innerHTML = '❌ Failed';
            showNotification('❌ Reporting failed: ' + (result.error || 'Unknown error'), 'error');
        }

    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = '❌ Error';
        showNotification('❌ Error: ' + error.message, 'error');
    }
}

// ==================== INITIALIZATION ====================

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Update stats immediately
    updateStats();

    // Set up periodic stats updates (fallback)
    setInterval(updateStats, 30000); // Every 30 seconds

    console.log('🚀 Urban Monitoring Platform Real-Time Client Initialized');
});

// ==================== DEBUG FUNCTIONS ====================

function testWebSocket() {
    console.log('Testing WebSocket connection...');
    console.log('Socket connected:', socket.connected);
    console.log('Socket ID:', socket.id);

    // Test vehicle simulation
    socket.emit('start_vehicle_simulation');

    // Test custom event
    socket.emit('test_event', {message: 'Hello from frontend'});
}

// Add test event listener
socket.on('test_response', function(data) {
    console.log('Test response:', data);
});

// Call this function from browser console to debug
window.testWebSocket = testWebSocket;

// ==================== OSM PARKING & FUEL FUNCTIONS (FREE APIs) ====================

async function loadOSMParkingSpots(lat, lon) {
    try {
        showNotification('🔄 Loading parking from OpenStreetMap...', 'info', 2000);

        // Use 500m radius to reduce API load and get more relevant results
        const response = await fetch(
            `${API_BASE}/osm/parking?lat=${lat}&lon=${lon}&radius=500`
        );
        const data = await response.json();

        layers.parking.clearLayers();

        if (data.spots && data.spots.length > 0) {
            data.spots.forEach(spot => {
                // Use blue OSM icon for all OSM parking spots
                const icon = icons.parking_osm;

                const marker = L.marker([spot.location.lat, spot.location.lon], { icon })
                    .addTo(layers.parking);

                marker.bindPopup(`
                    <strong>🅿️ ${spot.name}</strong><br>
                    Type: ${spot.parking_type || 'Unknown'}<br>
                    Capacity: ${spot.capacity || 'Unknown'}<br>
                    Fee: ${spot.fee || 'Unknown'}<br>
                    Access: ${spot.access || 'Public'}<br>
                    <br>
                    <small style="color: #3498db;">📍 Source: OpenStreetMap</small>
                    <br><br>
                    <button onclick="calculateRouteToAmenity(${spot.location.lat}, ${spot.location.lon}, '${spot.name.replace(/'/g, "\\'")}', 'parking')" 
                        style="width: 100%; background: #3742fa; color: white; border: none; padding: 8px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                        🗺️ Show Route Here
                    </button>
                `);
            });

            map.addLayer(layers.parking);

            // Update location marker popup
            if (amenityLocationState.marker) {
                amenityLocationState.marker.setPopupContent(
                    `<b>📍 Search Location</b><br>Found ${data.count} parking spots nearby`
                );
            }

            showNotification(`✅ Found ${data.count} parking areas within 500m`, 'success');
        } else {
            // Update location marker popup
            if (amenityLocationState.marker) {
                amenityLocationState.marker.setPopupContent(
                    `<b>📍 Search Location</b><br>No parking found nearby`
                );
            }

            showNotification('ℹ️ No parking found in this area. Try another location.', 'info');
        }

    } catch (error) {
        console.error('Error loading OSM parking:', error);
        showNotification('❌ Failed to load parking data', 'error');
    }
}

async function loadOSMGasStations(lat, lon) {
    try {
        showNotification('🔄 Loading gas stations from OpenStreetMap...', 'info', 2000);

        // Use 2km radius for gas stations (they're less frequent than parking)
        const response = await fetch(
            `${API_BASE}/osm/gas-stations?lat=${lat}&lon=${lon}&radius=2000`
        );
        const data = await response.json();

        layers.gasStations.clearLayers();

        if (data.stations && data.stations.length > 0) {
            data.stations.forEach(station => {
                const marker = L.marker([station.location.lat, station.location.lon], {
                    icon: icons.gas_station
                }).addTo(layers.gasStations);

                marker.bindPopup(`
                    <strong>⛽ ${station.name}</strong><br>
                    ${station.brand ? `Brand: ${station.brand}<br>` : ''}
                    ${station.address ? `Address: ${station.address}<br>` : ''}
                    <br>
                    <small style="color: #e74c3c;">📍 Source: OpenStreetMap</small>
                    <br><br>
                    <button onclick="calculateRouteToAmenity(${station.location.lat}, ${station.location.lon}, '${station.name.replace(/'/g, "\\'")}', 'gas')" 
                        style="width: 100%; background: #e74c3c; color: white; border: none; padding: 8px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                        🗺️ Show Route Here
                    </button>
                `);
            });

            map.addLayer(layers.gasStations);

            // Update location marker popup
            if (amenityLocationState.marker) {
                amenityLocationState.marker.setPopupContent(
                    `<b>📍 Search Location</b><br>Found ${data.count} gas stations nearby`
                );
            }

            showNotification(`✅ Found ${data.count} gas stations within 2km`, 'success');
        } else {
            if (amenityLocationState.marker) {
                amenityLocationState.marker.setPopupContent(
                    `<b>📍 Search Location</b><br>No gas stations found nearby`
                );
            }
            showNotification('ℹ️ No gas stations found. Try another location.', 'info');
        }

    } catch (error) {
        console.error('Error loading OSM gas stations:', error);
        showNotification('❌ Failed to load gas stations', 'error');
    }
}

async function loadOSMEVChargingStations(lat, lon) {
    try {
        showNotification('🔄 Loading EV charging stations from OpenStreetMap...', 'info', 2000);

        // Use 2km radius for EV charging stations
        const response = await fetch(
            `${API_BASE}/osm/ev-charging?lat=${lat}&lon=${lon}&radius=2000`
        );
        const data = await response.json();

        layers.evCharging.clearLayers();

        if (data.stations && data.stations.length > 0) {
            data.stations.forEach(station => {
                const marker = L.marker([station.location.lat, station.location.lon], {
                    icon: icons.ev_charging
                }).addTo(layers.evCharging);

                marker.bindPopup(`
                    <strong>🔌 ${station.name}</strong><br>
                    ${station.operator ? `Operator: ${station.operator}<br>` : ''}
                    ${station.network ? `Network: ${station.network}<br>` : ''}
                    Capacity: ${station.capacity || 'Unknown'}<br>
                    Fee: ${station.fee || 'Unknown'}<br>
                    <br>
                    <small style="color: #27ae60;">📍 Source: OpenStreetMap</small>
                    <br><br>
                    <button onclick="calculateRouteToAmenity(${station.location.lat}, ${station.location.lon}, '${station.name.replace(/'/g, "\\'")}', 'ev')" 
                        style="width: 100%; background: #27ae60; color: white; border: none; padding: 8px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                        🗺️ Show Route Here
                    </button>
                `);
            });

            map.addLayer(layers.evCharging);

            // Update location marker popup
            if (amenityLocationState.marker) {
                amenityLocationState.marker.setPopupContent(
                    `<b>📍 Search Location</b><br>Found ${data.count} EV charging stations nearby`
                );
            }

            showNotification(`✅ Found ${data.count} EV charging stations within 2km`, 'success');
        } else {
            if (amenityLocationState.marker) {
                amenityLocationState.marker.setPopupContent(
                    `<b>📍 Search Location</b><br>No EV charging stations found nearby`
                );
            }
            showNotification('ℹ️ No EV charging found. Try another location.', 'info');
        }

    } catch (error) {
        console.error('Error loading OSM EV stations:', error);
        showNotification('❌ Failed to load EV charging stations', 'error');
    }
}

