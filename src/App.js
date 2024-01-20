import React from "react";
import Carousel3D from "./componenets/Carousel3D";
import "./index.css";
import { schedules } from "./data/schedule"; // Your data file
import Header from "./Header";
import Footer from "./Footer";
import "./index.js";

const App = () => {
  return (
    <div className="App">
      <Header />
      <Carousel3D schedules={schedules} />
      <Footer />
    </div>
  );
};

export default App;
