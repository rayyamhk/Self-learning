const MongoClient = require('mongodb').MongoClient;
const logger = require('./logger');

let db = null;

const collections = {},
      collectionNames = ['Users', 'Comments'];

async function initDatabase() {
  try {
    const {
      MONGO_USERNAME,
      MONGO_PASSWORD,
      MONGO_DATABASE_NAME,
    } = process.env;
    const configs = {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    };
    const uri = `mongodb+srv://${MONGO_USERNAME}:${MONGO_PASSWORD}@auth.rzltx.mongodb.net/${MONGO_DATABASE_NAME}?retryWrites=true&w=majority`;
    const client = new MongoClient(uri, configs);

    await client.connect();
    db = await client.db(MONGO_DATABASE_NAME);
    logger.info('MongoDB connection success!');
    return db;
  } catch (err) {
    throw err;
  }
};

async function getCollection(key) {
  if (!collectionNames.includes(key)) {
    throw new Error('Collection does not exist');
  }

  if (collections[key]) {
    return collections[key];
  }

  try {
    if (!db) {
      await initDatabase();
    }
    const collection = db.collection(key);
    collections[key] = collection;
    return collection;
  } catch (err) {
    throw err;
  }
}

module.exports = getCollection;
