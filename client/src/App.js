import './App.css';
import 'mapbox-gl/dist/mapbox-gl.css';
import { useState, useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import Police from './images/police.png';
import Ambulance from './images/ambulance.png';
import Firetruck from './images/firetruck.png';
import Drone from './images/drone.png';

function App() {

  const [date, setDate] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setDate(new Date());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const format = {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  };

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  mapboxgl.accessToken = process.env.REACT_APP_MAPBOXGL_API_KEY;

  const URBANA_CENTER = [-88.211105, 40.113159];

  useEffect(() => {
    if (mapRef.current) return;

    mapRef.current = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/streets-v11',
      center: URBANA_CENTER,
      zoom: 13
    });
  }, []);

  const POINTS = [
  { id: 1, lngLat: [-88.211105, 40.113159], title: 'Downtown' },
  { id: 2, lngLat: [-88.2433, 40.1122], title: 'Incident' },
];

useEffect(() => {
  if (!mapRef.current) return;

  const markers = POINTS.map((pt) => {
    const el = document.createElement('div');
    el.className = 'red flash large';
    el.title = pt.title;

    return new mapboxgl.Marker(el)
      .setLngLat(pt.lngLat)
      .addTo(mapRef.current);
  });

  return () => markers.forEach(m => m.remove());
}, []);

  return (
    <div id="app">
      <header>
        <div id="branding">
          <img/>
          <p>HackIllinois</p>
        </div>
        <p>{date.toLocaleString("en-US", format).replace(/\bat\b/g, "\u00a0")}</p>
      </header>
      <div id="content">
        <div id="main">
          <div id="map">
            <div
              ref={mapContainerRef}
              style={{ height: '100%', width: '100%' }}
            />
          </div>
          <div id="calls">
            {[0, 0, 0, 0, 0, 0, 0].map((zero, index) => (
              <div id="call">
                <p id="number">(999) 999-9999</p>
                <p id="severity" className="red red-bg"><span className="red flash"/>Severity</p>
                <p id="situation">Situation<span id="address"> - Address</span></p>
                <p id="transcript">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
              </div>
            ))}
          </div>
        </div>
        <div id="info">
          <div id="ticket">
            <h3>Incident Ticket - ID</h3>
            <p className="title">Caller Info</p><br/>
            <p className="subtitle">Name:</p>
            <p className="info">John Doe</p><br/>
            <p className="subtitle">Phone:</p>
            <p className="info">(217) 555-0192</p>
            <br/>
            <p className="title">Location</p><br/>
            <p className="subtitle">Address:</p>
            <p className="info">123 Main St, Apt 4B</p><br/>
            <p className="subtitle">Coordinates:</p>
            <p className="info">40.1122, -88.2433</p>
            <br/>
            <p className="title">Incident Details</p><br/>
            <p className="subtitle">Type:</p>
            <p className="info">Robbery</p><br/>
            <p className="subtitle">Severity:</p>
            <p className="info">Urgent</p><br/>
            <p className="subtitle">Notes:</p>
            <p className="info">Armed Suspect, No Injuries</p>
            <br/>
            <p className="title">Status</p><br/>
            <p className="subtitle">Current Status:</p>
            <p className="info">En Route</p><br/>
            <p className="subtitle">Resource Assignment:</p>
            <p className="info">Police</p><br/>
          </div>
          <div id="actions">
            <button id="police">
              <img src={Police}/>
              <br/>
              Police
            </button>
            <button id="ambulance">
              <img src={Ambulance}/>
              <br/>
              Ambulance
            </button>
            <button id="firetruck">
              <img src={Firetruck}/>
              <br/>
              Fire Truck
            </button>
            <button id="drone">
              <img src={Drone}/>
              <br/>
              Drone Unit
            </button>
          </div>
          <div id="extended-transcript">
            <h3>Call Transcript</h3>
            <div class="content">
              {[0, 0, 0, 0, 0, 0, 0].map((zero, index) => (
                <div>
                  <div class="chat-entry">
                    <span class="speaker">Caller:</span>
                    <span class="message">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</span>
                  </div>
                  <div class="chat-entry">
                    <span class="speaker">Dispatcher:</span>
                    <span class="message">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
