require("dotenv").config();
const express = require("express");
const { MongoClient } = require("mongodb");
const cors = require("cors");

const app = express();
const port = process.env.PORT || 3001;

const userName = process.env.MONGO_USERNAME;
const password = process.env.MONGO_PASSWORD;
const uri = `mongodb+srv://${userName}:${password}@cluster0.ml8jvfc.mongodb.net/?retryWrites=true&w=majority`;

let db;

// Initialize MongoDB Connection once
MongoClient.connect(uri, { useUnifiedTopology: true })
  .then((client) => {
    console.log("Connected to Database");
    db = client.db("withTheSpread");
  })
  .catch((error) => console.error(error));

app.use(cors());

app.get("/getData", async (req, res) => {
  try {
    const collection = db.collection("season_spreads");
    const data = await collection.find().toArray();
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).send("Error fetching data");
  }
});

app.listen(port, () => {
  console.log(`Server not running on port ${port}`);
});
