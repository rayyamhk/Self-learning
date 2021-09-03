const { logger } = require('../utils');
const { createUser: addUser } = require('../utils/Users');

async function createUser(req, res) {
  try {
    const {
      username,
      email,
      password,
    } = req.body;

    const {
      statusCode,
      message,
    } = await addUser(username, email, password);
    logger.info(message);
    return res.status(statusCode).send(message).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = createUser;
