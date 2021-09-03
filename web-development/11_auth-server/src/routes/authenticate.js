const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { getUser, updateUser } = require('../utils/Users');
const { logger } = require('../utils');

async function authenticate(req, res) {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      logger.warn('Missing data');
      return res.status(400).send('Missing data').end();
    }

    const { status, statusCode, message, payload: user } = await getUser(email);

    if (!status) {
      logger.warn(message);
      return res.status(statusCode).send('Incorrect email or password').end();
    }

    if (user.isBlocked) {
      logger.warn('Account has been suspended');
      return res.status(403).send('Account has been suspended').end();
    }

    const isValid = await bcrypt.compare(password, user.password);
    if (isValid) {
      const payload = {
        username: user.username,
        email: user.email,
        avatar: user.avatar,
        role: user.role,
      };
      const accessToken = await jwt.sign(payload, process.env.JWT_ACCESS_TOKEN_KEY, { expiresIn: process.env.JWT_ACCESS_TOKEN_EXPIRED });
      const refreshToken = await jwt.sign(payload, process.env.JWT_REFRESH_TOKEN_KEY, { expiresIn: process.env.JWT_REFRESH_TOKEN_EXPIRED });

      await updateUser(email, {
        lastLogin: new Date().toISOString(),
        loginAttempts: 0,
        refreshToken,
      });

      logger.info(`${user.email}: Log in successfully`);
      return res.status(200).json({
        user: payload,
        accessToken,
        refreshToken,
      }).end();
    }

    if (user.loginAttempts === 4) {
      await updateUser(email, {
        loginAttempts: 5,
        isBlocked: true,
      });
      logger.warn('Account has been suspended, too many login failures');
      return res.status(403).send('Account has been suspended, too many login failures').end();
    }

    await updateUser(email, { loginAttempts: user.loginAttempts + 1 });
    logger.warn('Incorrect password');
    return res.status(400).send('Incorrect email or password').end();

  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = authenticate;
