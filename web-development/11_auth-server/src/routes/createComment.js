const { logger } = require('../utils');
const { createComment: addComment } = require('../utils/Comments');

async function createComment(req, res) {
  try {
    const {
      postId,
      user,
      body,
    } = req.body;

    const {
      statusCode,
      message,
    } = await addComment(postId, user, body);
    logger.info(message);
    return res.status(statusCode).send(message).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = createComment;
