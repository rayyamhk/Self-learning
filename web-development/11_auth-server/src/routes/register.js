const {
  sendMail,
  genVerification,
  Cache,
  logger,
} = require('../utils');
const { createUser, updateUser, getUser } = require('../utils/Users');

const cache = new Cache({
  ttl: 5 * 60 * 1000, // 5m
  period: 60 * 60 * 1000, // 1h
});

async function register(req, res) {
  const { task } = req.query;

  if (task === 'verify') {
    const { verificationCode, email } = req.body;
    const item = cache.get(email);

    if (!verificationCode || !item) {
      return res.status(400).send('Bad request, you can try to resend the verification code').end();
    }

    if (verificationCode !== item) {
      return res.status(403).send('Verification code does not match').end();
    }

    cache.remove(email);

    try {
      const { status, statusCode, message } = await updateUser(email, { isActivated: true });
      if (status) {
        logger.info(`User with email ${email} has been activated`);
        return res.status(200).send('Activated successfully!').end();
      }
      logger.error(message);
      return res.status(statusCode).send(message).end();
    } catch (err) {
      logger.error(err);
      return res.status(500).send('500 Internal Server Error').end();
    }
  }

  if (task === 'resend') {
    const { email } = req.body;
    try {
      const { status, statusCode, message, payload: user } = await getUser(email);
      if (status) {
        if (user.isActivated) {
          logger.warn(`User with email ${email} has already been activated!`);
          return res.status(400).send('Bad request').end();
        }
        const verificationCode = genVerification();
        cache.set(email, verificationCode);
        await sendMail({
          to: `User <${email}>`,
          subject: 'Activate your account',
          text: `Your verification code: ${verificationCode}, please verify it within 5 minutes.`,
        });
        logger.info(`Email has been sent to ${email}, please verify it within 5 minutes.`);
        return res.status(200).send(`An email has been sent to your address, please verify it within 5 minutes.`).end();
      }
      logger.error(message);
      return res.status(statusCode).send('Bad request').end();
    } catch (err) {
      logger.error(err);
      return res.status(500).send('500 Internal Server Error').end();
    }
  }

  const {
    username,
    email,
    password,
  } = req.body;

  try {
    const { status, statusCode, message } = await createUser(username, email, password);
    if (status) {
      const verificationCode = genVerification();
      cache.set(email, verificationCode);
      await sendMail({
        to: `User <${email}>`,
        subject: 'Activate your account',
        text: `Your verification code: ${verificationCode}, please verify it within 5 minutes.`,
      });
      logger.info(`Email has been sent to ${email}`);
      return res.status(200).send(`Register successfully! An email has been sent to your address, please verify it within 5 minutes.`).end();
    }
    logger.error(message);
    return res.status(statusCode).send(message).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = register;