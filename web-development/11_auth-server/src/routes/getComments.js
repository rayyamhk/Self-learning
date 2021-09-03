const { logger, optionsConstructor } = require('../utils');
const { getComments: getAllComments } = require('../utils/Comments');

async function getComments(req, res) {
  try {
    const { postId } = req.query;

    const options = optionsConstructor(req.query);

    const query = {};
    if (postId) {
      query.postId = postId;
    }

    const { statusCode, message, payload } = await getAllComments(query, options);
    logger.info(message);
    return res.status(statusCode).json({ comments: payload }).end();
  } catch (err) {
    logger.error(err);
    return res.status(500).send('500 Internal Server Error').end();
  }
};

module.exports = getComments;
