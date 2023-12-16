// Desc: Group schedules by week
const groupByWeek = (schedules) => {
  const grouped = {};
  schedules.forEach((schedule) => {
    if (!grouped[schedule.week]) {
      grouped[schedule.week] = [];
    }
    grouped[schedule.week].push(schedule);
  });
  return grouped;
};

export default groupByWeek;
