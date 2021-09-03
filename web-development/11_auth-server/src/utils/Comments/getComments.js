const getCollection = require('../getCollection');

async function getComments(query, options) {
  try {
    const Comments = await getCollection('Comments');
    const comments = await Comments.find(query, options).toArray();
    return {
      status: true,
      statusCode: 200,
      message: 'Retrieve successfully',
      payload: comments,
    };
  } catch (err) {
    throw err;
  }
};

module.exports = getComments;
