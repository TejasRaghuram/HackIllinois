import './App.css';
import { useState, useEffect } from 'react';

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
          <div id="map"></div>
          <div id="calls"></div>
        </div>
        <div id="info"></div>
      </div>
    </div>
  );
}

export default App;
