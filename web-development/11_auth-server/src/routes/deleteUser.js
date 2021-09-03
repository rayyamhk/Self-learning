const { logger } = require('../utils');
const { deleteUser: removeUser } = require('../utils/Users');

async function deleteUser(req, res) {
  try {
    const { email } = req.body;

    const {
      statusCode,
      message,
    } = await removeUser(email);
    logger.info(message);
    return res.status(statusCode).send(message).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = deleteUser;
