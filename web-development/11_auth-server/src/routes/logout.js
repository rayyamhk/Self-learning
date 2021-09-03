const { logger } = require('../utils');
const { updateUser } = require('../utils/Users');

async function logout(req, res) {
  try {
    const { email } = req.body;

    const {
      status,
      statusCode,
      message,
    } = await updateUser(email, { refreshToken: null });
  
    if (status) {
      logger.info(`${email}: Logged out successfully`);
      return res.status(statusCode).send('Logged out successfully').end();
    }

    logger.warn(message);
    return res.status(statusCode).send(message).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error');
  }
};

module.exports = logout;
