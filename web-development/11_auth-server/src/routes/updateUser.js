const { logger } = require('../utils');
const { updateUser: modifyUser } = require('../utils/Users');

async function updateUser(req, res) {
  try {
    const {
      email,
      updates,
    } = req.body;

    const {
      username,
      password,
      avatar,
      age,
      gender,
      bio,
      address,
      role,
    } = updates;

    const {
      statusCode,
      message,
    } = await modifyUser(email, {
      username,
      password,
      avatar,
      age,
      gender,
      bio,
      address,
      role,
    });

    return res.status(statusCode).send(message).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = updateUser;
