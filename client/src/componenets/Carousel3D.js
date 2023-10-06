import React, { useState, useRef, useEffect } from "react";
import "../index.css";
import groupByWeek from "./groupByWeek";

const Carousel3D = ({ schedules }) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const groupedSchedules = groupByWeek(schedules);
  const carouselRef = useRef(null);
  let touchStartX = 0;
  const audio = new Audio(process.env.PUBLIC_URL + "/iphone_click.mp3");

  useEffect(() => {
    audio.addEventListener("timeupdate", function () {
      if (audio.currentTime >= 0.3) {
        audio.pause();
        audio.currentTime = 0;
      }
    });
  }, []);

  const playClickSound = () => {
    audio.currentTime = 0;
    audio.play();
  };

  const vibrateDevice = () => {
    if (navigator.vibrate) {
      navigator.vibrate(100);
    }
  };

  const rotateCarousel = (direction) => {
    playClickSound();
    vibrateDevice();
    const totalWeeks = Object.keys(groupedSchedules).length;

    if (direction === "left") {
      setSelectedIndex(
        (prevIndex) => (prevIndex - 1 + totalWeeks) % totalWeeks
      );
    } else {
      setSelectedIndex((prevIndex) => (prevIndex + 1) % totalWeeks);
    }
  };

  const handleTouchStart = (e) => {
    touchStartX = e.touches[0].clientX;
  };

  const handleTouchEnd = (e) => {
    const touchEndX = e.changedTouches[0].clientX;
    if (touchEndX > touchStartX + 30) {
      rotateCarousel("left");
    } else if (touchEndX < touchStartX - 30) {
      rotateCarousel("right");
    }
  };

  const [seasonSpreads, setSeasonSpreads] = useState([]);

  useEffect(() => {
    fetch("http://localhost:3001/getData")
      .then((response) => response.json())
      .then((data) => setSeasonSpreads(data))
      .catch((error) => console.error("Error fetching season spreads:", error));
  }, []);

  return (
    <div
      className="carousel-container"
      ref={carouselRef}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      aria-label="3D Carousel"
    >
      <div className="carousel">
        {Object.keys(groupedSchedules).map((week, index) => (
          <div
            key={index}
            className={`carousel-item ${
              index === selectedIndex ? "active" : ""
            }`}
            style={{
              transform: `rotateY(${
                (index - selectedIndex) * 40
              }deg) translateZ(300px)`,
            }}
            onClick={() => rotateCarousel("right")}
            aria-label={`Week ${week} Carousel Item`}
          >
            <h2 className="carousel-item-title">{`Week ${week}`}</h2>

            <div className="scrollable-container">
              <table className="carousel-item-table">
                <thead>
                  <tr>
                    <th>Home</th>
                    <th>Moneyline</th>
                    <th>Spread</th>
                    <th>Moneyline</th>
                    <th>Away</th>
                  </tr>
                </thead>
                <tbody>
                  {groupedSchedules[week].map((game, i) => {
                    const spreadData = seasonSpreads.find(
                      (spread) => spread.opponent === game.home_team
                    );
                    return (
                      <tr key={i}>
                        <td>
                          <img
                            src={game.home_logo}
                            alt={`${game.home_team} logo`}
                          />{" "}
                          {game.home_team}
                        </td>
                        <td>{game.home_moneyline}</td>
                        <td>{spreadData ? spreadData.spread : "N/A"}</td>
                        <td>{game.away_moneyline}</td>
                        <td>
                          <img
                            src={game.away_logo}
                            alt={`${game.away_team} logo`}
                          />{" "}
                          {game.away_team}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
      <div className="carousel-indicators">
        {Object.keys(groupedSchedules).map((week, index) => (
          <button
            key={index}
            className={`carousel-indicator ${
              index === selectedIndex ? "active" : ""
            }`}
            onClick={() => setSelectedIndex(index)}
          >
            {week}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Carousel3D;
