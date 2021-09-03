const { logger } = require('../utils');
const { getUser: getSingleUser } = require('../utils/Users');

async function getUser(req, res) {
  try {
    const { email } = req.query;

    const {
      statusCode,
      message,
      payload,
    } = await getSingleUser(email);

    const {
      username,
      avatar,
      age,
      gender,
      bio,
      address,
      role,
      createdAt,
      updatedAt,
      lastLogin,
      isActivated,
    } = payload;

    logger.info(message);
    return res.status(statusCode).send({
      username,
      email,
      avatar,
      age,
      gender,
      bio,
      address,
      role,
      createdAt,
      updatedAt,
      lastLogin,
      isActivated,
    }).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = getUser;
