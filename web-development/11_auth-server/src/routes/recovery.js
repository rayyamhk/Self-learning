const {
  sendMail,
  logger,
} = require('../utils');
const { updateUser, getUser } = require('../utils/Users');
const jwt = require('jsonwebtoken');

async function recovery(req, res) {
  const { task } = req.query;

  if (task === 'reset') {
    const { token, password } = req.body;

    try {
      const { email, ip } = await jwt.verify(token, process.env.JWT_VERIFY_TOKEN_KEY);
      logger.info(`Request ip address: ${ip}`);
      if (ip !== req.requestIp) {
        logger.error('Ip address does not match');
        return res.status(403).send('Unauthorized').end();
      }

      const { status, statusCode, message } = await updateUser(email, { password });
      if (status) {
        logger.info(message);
        return res.status(statusCode).send('Your password has been changed successfully').end();
      }
      logger.warn(message);
      return res.status(400).send('Bad request').end();

    } catch (err) {
      if (err.name === 'TokenExpiredError') {
        logger.error('Verify token expired');
        return res.status(403).send('Unauthorized: Request expired');
      }
      if (err.name === 'JsonWebTokenError') {
        logger.error('Incorrect verify token');
        return res.status(403).send('Unauthorized').end();
      }
      logger.error(err);
      return res.status(500).send('500 Internal Server Error').end();
    }
  } else {
    const { email } = req.body;

    if (!email) {
      return res.status(400).send('Bad request').end();
    }
    
    try {
      const { status, message } = await getUser(email);
      if (status) {
        const payload = { email, ip: req.requestIp };
        const verifyToken = await jwt.sign(payload, process.env.JWT_VERIFY_TOKEN_KEY, { expiresIn: '5m' });
        await sendMail({
          to: `User <${email}>`,
          subject: 'Reset your password',
          text: `Please verify your password recovery request via this url: ${process.env.PRODUCT_URL}/recovery/reset?token=${verifyToken}`,
        });
        logger.info(`Email has been sent to ${email}`);
        logger.info(`Request ip address: ${req.requestIp}`);
        return res.status(200).send(`An email has been sent to your address, please verify it within 5 minutes.`).end();
      }
      logger.warn(message);
      return res.status(400).send('Bad request').end();
    } catch (err) {
      logger.error(err);
      return res.status(500).send('500 Internal Server Error').end();
    }
  }
};

module.exports = recovery;
