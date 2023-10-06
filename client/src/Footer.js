import React from "react";
import "./index.css";

const Footer = () => {
  return (
    <div className="footer">
      <div className="footer-container">
        <div className="footer-column">
          <h4 className="footer-title">Column 1</h4>
          <ul className="footer-list">
            <li className="footer-list-item">
              <a href="#">Item 1</a>
            </li>
            <li className="footer-list-item">
              <a href="#">Item 2</a>
            </li>
          </ul>
        </div>
        <div className="footer-column">
          <h4 className="footer-title">Column 2</h4>
          <ul className="footer-list">
            <li className="footer-list-item">
              <a href="#">Item 1</a>
            </li>
            <li className="footer-list-item">
              <a href="#">Item 2</a>
            </li>
          </ul>
        </div>
        {/* Add more columns as needed */}
      </div>
      <div className="copy">
        <p>&copy; 2023 Todder & Co. All rights reserved.</p>
      </div>
    </div>
  );
};

export default Footer;
