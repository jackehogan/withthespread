import React from "react";
import Carousel3D from "./componenets/Carousel3D";
import "./index.css";
import { schedules } from "./data/data"; // Your data file
import Header from "./Header";
import Footer from "./Footer";

const App = () => {
  return (
    <div className="App">
      <Header />
      <h1>With The Spread</h1>
      <Carousel3D schedules={schedules} />
      <Footer />
    </div>
  );
};

export default App;
