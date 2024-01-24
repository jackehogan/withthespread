import React, { useState, useRef, useEffect } from "react";
import "../index.css";
import groupByWeek from "./groupByWeek";
import Advertisement from "./advertisement";

const Carousel3D = ({ schedules }) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const groupedSchedules = groupByWeek(schedules);
  const carouselRef = useRef(null);
  let touchStartX = 0;

  const audioRef = useRef(new Audio(process.env.PUBLIC_URL + "/iphone_click.mp3"));

  useEffect(() => {
    const audio = audioRef.current;
    const handleTimeUpdate = () => {
      if (audio.currentTime >= 0.3) {
        audio.pause();
        audio.currentTime = 0;
      }
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
    };
  }, []);

  const playClickSound = () => {
    const audio = audioRef.current;
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
    // Logic to find the closest week
    // ...
    setSelectedIndex(/* calculation based on schedules */);
  }, [schedules]);

  return (
    <div
      className="carousel-container"
      ref={carouselRef}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      aria-label="3D Carousel"
    >
      {/* Render carousel items */}
    </div>
  );
};

export default Carousel3D;

