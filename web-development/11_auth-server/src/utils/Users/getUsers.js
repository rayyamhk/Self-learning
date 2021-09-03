const getCollection = require('../getCollection');

async function getUsers(query, options) {
  try {
    const Users = await getCollection('Users');
    const users = await Users.find(query, options).toArray();
    return {
      status: true,
      statusCode: 200,
      message: 'Retrieve successfully',
      payload: users,
    };
  } catch (err) {
    throw err;
  }
};

module.exports = getUsers;
