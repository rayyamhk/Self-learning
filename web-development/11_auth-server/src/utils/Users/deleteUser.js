const getCollection = require('../getCollection');

async function deleteUser(email) {
  if (!email) {
    return {
      status: false,
      statusCode: 400,
      message: 'Missing data',
    };
  }

  try {
    const Users = await getCollection('Users');
    const result = await Users.deleteOne({ email });
    if (result.deletedCount === 1) {
      return {
        status: true,
        statusCode: 200,
        message: `User with email ${email} was deleted successfully!`,
      };
    }

    return {
      status: false,
      statusCode: 400,
      message: 'User does not exist',
    };
  } catch (err) {
    throw err;
  }
};

module.exports = deleteUser;
