import './App.css';
import 'mapbox-gl/dist/mapbox-gl.css';
import { useState, useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';

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
          <div id="calls"></div>
        </div>
        <div id="info"></div>
      </div>
    </div>
  );
}

export default App;
