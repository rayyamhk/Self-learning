const bcrypt = require('bcrypt');
const { getUser } = require('../utils/Users');
const { logger } = require('../utils');

async function validatePassword(req, res) {
  const email = req.params.email;
  console.log(req.params);
  const { password } = req.body;

  if (!email || !password) {
    logger.warn('Missing data');
    return res.status(400).send('Missing data').end();
  }

  try {
    const { status, statusCode, message, payload: user } = await getUser(email);

    if (!status) {
      logger.warn(message);
      return res.status(statusCode).send('Bad request').end();
    }

    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      logger.error('Incorrect password');
      return res.status(400).send('Incorrect password').end();
    }

    logger.info('Correct password');
    return res.status(200).send('Correct password').end();

  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = validatePassword;
