const getCollection = require('../getCollection');
const { ObjectID } = require('mongodb');

async function deleteComment(id) {
  if (!id) {
    return {
      status: false,
      statusCode: 400,
      message: 'Missing data',
    };
  }
  try {
    const Comments = await getCollection('Comments');
    const result = await Comments.deleteOne({ _id: new ObjectID(id) });
    if (result.deletedCount === 1) {
      return {
        status: true,
        statusCode: 200,
        message: 'Comment was deleted successfully!',
      };
    }
    return {
      status: false,
      statusCode: 400,
      message: 'Comment does not exist',
    };
  } catch (err) {
    throw err;
  }
};

module.exports = deleteComment;
