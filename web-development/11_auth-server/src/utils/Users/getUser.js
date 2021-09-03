const getCollection = require('../getCollection');

async function getUser(email) {
  if (!email) {
    return {
      status: false,
      statusCode: 400,
      message: 'Missing data',
    };
  }

  try {
    const Users = await getCollection('Users');
    const user = await Users.findOne({ email });
    if (user) {
      return {
        status: true,
        statusCode: 200,
        message: 'Retrieve successfully',
        payload: user,
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

module.exports = getUser;
