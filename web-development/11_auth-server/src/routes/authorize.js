const { authorize: authorizeUser } = require('../utils/Users');
const { logger } = require('../utils');

async function authorize(req, res) {
  const { email } = req.body;

  if (!email) {
    return res.status(400).send('Missing data').end();
  }

  try {
    const accessBearerToken = req.get('jwt-access-token');
    if (!accessBearerToken || accessBearerToken.split(' ').length !== 2) {
      logger.warn('Unauthorized: Missing access token');
      return res.status(403).send('Unauthorized').end();
    }

    const refreshBearerToken = req.get('jwt-refresh-token');
    if (!refreshBearerToken || refreshBearerToken.split(' ').length !== 2) {
      logger.warn('Unauthorized: Access token expired and missing refresh token');
      return res.status(403).send('Unauthorized').end();
    }

    const accessToken = accessBearerToken.split(' ')[1];
    const refreshToken = refreshBearerToken.split(' ')[1];

    const {
      status,
      statusCode,
      message,
      payload,
    } = await authorizeUser(email, accessToken, refreshToken);

    if (status) {
      logger.info(message);
      return res.status(statusCode).json(payload).end();
    }

    logger.warn(message);
    return res.status(statusCode).send('Unauthorized').end();

  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error');
  }
};

module.exports = authorize;
