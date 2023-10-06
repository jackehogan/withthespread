const CarouselItem = ({ week, games, selectedIndex, index, seasonSpreads }) => {
  return (
    <div
      className={`carousel-item ${index === selectedIndex ? "active" : ""}`}
      style={{
        transform: `rotateY(${
          (index - selectedIndex) * 40
        }deg) translateZ(300px)`,
      }}
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
            {games.map((game, i) => {
              const spreadData = seasonSpreads.find(
                (spread) => spread.opponent === game.home_team
              );
              return <tr key={i}>{/* ... (rest of the table row) */}</tr>;
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
