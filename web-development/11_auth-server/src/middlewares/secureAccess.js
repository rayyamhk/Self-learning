const { authorize } = require('../utils/Users');

async function secureAccess(req, res, next) {
  if (process.env.SECURE_ACCESS === 'false') {
    return next();
  }
  let email;
  if (req.body.email) {
    email = req.body.email;
  } else {
    email = req.query.email;
  }
  const accessBearerToken = req.get('jwt-access-token');
  const refreshBearerToken = req.get('jwt-refresh-token');

  if (!accessBearerToken || accessBearerToken.split(' ').length !== 2) {
    return res.status(403).send('Unauthorized').end();
  }

  if (!refreshBearerToken || refreshBearerToken.split(' ').length !== 2) {
    return res.status(403).send('Unauthorized').end();
  }

  const accessToken = accessBearerToken.split(' ')[1];
  const refreshToken = refreshBearerToken.split(' ')[1];

  const {
    status,
    statusCode,
    message, 
  } = await authorize(email, accessToken, refreshToken);

  if (status) {
    return next();
  }

  return res.status(statusCode).send(message).end();
};

module.exports = secureAccess;
