import React, { useState, useRef, useEffect } from "react";
import "../index.css";
import groupByWeek from "./groupByWeek";
import Advertisement from "./advertisement";

const Carousel3D = ({ schedules }) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const groupedSchedules = groupByWeek(schedules);
  const carouselRef = useRef(null);
  let touchStartX = 0;
  // eslint-disable-next-line to the line before.
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

  useEffect(() => {
    // Get the current date
    const currentDate = new Date();

    // Find the closest week to the current date
    const closestWeek = Object.keys(groupedSchedules).reduce(
      (closest, week) => {
        const weekDate = new Date(groupedSchedules[week][0].date);
        const timeDiff = Math.abs(weekDate - currentDate);
        if (timeDiff < closest.timeDiff) {
          return { week, timeDiff };
        }
        return closest;
      },
      { week: Object.keys(groupedSchedules)[0], timeDiff: Infinity }
    );

    // Set the selectedIndex to the index of the closest week
    // eslint-disable-next-line to the line before.
    setSelectedIndex(Object.keys(groupedSchedules).indexOf(closestWeek.week));
  }, [schedules]);

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
            onClick={() => rotateCarousel("right")}
            aria-label={`Week ${week} Carousel Item`}
          >
            <div className="carousel-header">
              <h2 className="carousel-item-title">{`Week ${week}`}</h2>
            </div>
            <table className="carousel-item-table">
              <thead>
                <tr>
                  <th>Teams</th>
                  <th>Spread</th>
                  <th>Score</th>
                  <th>Pred Spread</th>
                  <th>Spread Score</th>
                  <th>WTS Cover</th>
                </tr>
              </thead>
              <tbody>
                {groupedSchedules[week].map((game, i) => {
                  const spreadDataHome = seasonSpreads.find(
                    (spread) =>
                      spread.Week === game.week && spread.Team === game.home
                  );
                  const spreadDataAway = seasonSpreads.find(
                    (spread) =>
                      spread.Week === game.week && spread.Team === game.away
                  );
                  return (
                    <tr key={i}>
                      <td>
                        <div className="team-container">
                          <img
                            src={game.home_logo}
                            alt={`${game.home_team} logo`}
                            className="team-logo"
                          />
                          {game.home}
                        </div>
                        <div></div>
                        <div className="team-container">
                          <img
                            src={game.away_logo}
                            alt={`${game.away_team} logo`}
                            className="team-logo"
                          />
                          {game.away}
                        </div>
                      </td>
                      <td>
                        <div className="stats">
                          <div className="numbers-container">
                            {spreadDataHome ? (
                              spreadDataHome.spread > 0 ? (
                                <span className="sign-positive">+</span>
                              ) : (
                                <span className="sign-negative">-</span>
                              )
                            ) : null}
                            {spreadDataHome
                              ? Math.abs(spreadDataHome.spread)
                              : "Coming Soon"}
                          </div>
                        </div>
                        <div className="stats">
                          <div className="numbers-container">
                            {spreadDataAway ? (
                              spreadDataAway.spread > 0 ? (
                                <span className="sign-positive">+</span>
                              ) : (
                                <span className="sign-negative">-</span>
                              )
                            ) : null}
                            {spreadDataAway
                              ? Math.abs(spreadDataAway.spread)
                              : ""}
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="stats">
                          {spreadDataHome
                            ? spreadDataHome.score
                            : "Coming Soon"}
                        </div>
                        <div className="stats">
                          {spreadDataAway ? spreadDataAway.score : ""}
                        </div>
                      </td>
                      <td>
                        <div className="stats">
                          <div className="numbers-container">
                            {spreadDataHome ? (
                              spreadDataHome.predspread > 0 ? (
                                <span className="sign-positive">+</span>
                              ) : (
                                <span className="sign-negative">-</span>
                              )
                            ) : null}
                            {spreadDataHome
                              ? Math.abs(
                                  Math.round(spreadDataHome.predspread * 10) /
                                    10
                                )
                              : "Coming Soon"}
                          </div>
                        </div>
                        <div className="stats">
                          <div className="numbers-container">
                            {spreadDataAway ? (
                              spreadDataAway.predspread > 0 ? (
                                <span className="sign-positive">+</span>
                              ) : (
                                <span className="sign-negative">-</span>
                              )
                            ) : null}
                            {spreadDataAway
                              ? Math.abs(
                                  Math.round(spreadDataAway.predspread * 10) /
                                    10
                                )
                              : ""}
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="stats">
                          <div className="numbers-container">
                            {spreadDataHome ? (
                              spreadDataHome.spreadscore > 0 ? (
                                <span className="sign-positive">+</span>
                              ) : (
                                <span className="sign-negative">-</span>
                              )
                            ) : null}
                            {spreadDataHome
                              ? Math.abs(
                                  Math.round(spreadDataHome.spreadscore * 10) /
                                    10
                                )
                              : "Coming Soon"}
                          </div>
                        </div>
                        <div className="stats">
                          <div className="numbers-container">
                            {spreadDataAway ? (
                              spreadDataAway.spreadscore > 0 ? (
                                <span className="sign-positive">+</span>
                              ) : (
                                <span className="sign-negative">-</span>
                              )
                            ) : null}
                            {spreadDataAway
                              ? Math.abs(
                                  Math.round(spreadDataAway.spreadscore * 10) /
                                    10
                                )
                              : ""}
                          </div>
                        </div>
                      </td>
                      <td>
                        <div>
                          <div className="stats">
                            <div className="numbers-container">
                              {spreadDataHome
                                ? Math.abs(
                                    Math.round(
                                      ((spreadDataHome.predspread -
                                        spreadDataHome.spreadscore) *
                                        10) /
                                        10
                                    )
                                  )
                                : null}
                            </div>
                          </div>
                        </div>
                        <div>
                          <div className="stats">
                            <div className="numbers-container">
                              {spreadDataAway
                                ? Math.abs(
                                    Math.round(
                                      ((spreadDataAway.predspread -
                                        spreadDataAway.spreadscore) *
                                        10) /
                                        10
                                    )
                                  )
                                : null}
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  );
                })}
                <Advertisement />
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Carousel3D;
