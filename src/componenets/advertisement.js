import React from "react";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import "../index.css";

const IframeCarousel = () => {
  const iframeUrls = [
    "https://mediaserver.betmgmpartners.com/renderBanner.do?zoneId=1703291&t=f&v=1&securedDomain=y",
    "https://mediaserver.betmgmpartners.com/renderBanner.do?zoneId=1703291&t=f&v=1&securedDomain=y",
    "https://mediaserver.betmgmpartners.com/renderBanner.do?zoneId=1703291&t=f&v=1&securedDomain=y",
    "https://mediaserver.betmgmpartners.com/renderBanner.do?zoneId=1703291&t=f&v=1&securedDomain=y",
    "https://mediaserver.betmgmpartners.com/renderBanner.do?zoneId=1703291&t=f&v=1&securedDomain=y",
    "https://mediaserver.betmgmpartners.com/renderBanner.do?zoneId=1703291&t=f&v=1&securedDomain=y",

    // Add more iframe URLs as needed
  ];

  const settings = {
    infinite: true,
    speed: 500, // Transition speed in milliseconds
    slidesToShow: 2, // Number of slides to show at once
    slidesToScroll: 1, // Number of slides to scroll at once
    autoplay: true, // Enable autoplay
    autoplaySpeed: 5000, // Autoplay interval in milliseconds
  };

  return (
    <div className="iframe-carousel">
      <Slider {...settings}>
        {iframeUrls.map((url, index) => (
          <div key={index}>
            <iframe
              title={`iframe-${index}`}
              src={url}
              frameBorder="0"
              marginHeight="0"
              marginWidth="0"
              scrolling="no"
              width="728"
              height="90"
            ></iframe>
          </div>
        ))}
      </Slider>
    </div>
  );
};

export default IframeCarousel;
