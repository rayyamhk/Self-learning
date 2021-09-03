const { logger, optionsConstructor } = require('../utils');
const { getUsers: getAllUsers } = require('../utils/Users');

async function getUsers(req, res) {
  try {
    const options = optionsConstructor(req.query);
    const {
      statusCode,
      message,
      payload,
    } = await getAllUsers({}, options);
    logger.info(message);
    return res.status(statusCode).json(payload).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = getUsers;
