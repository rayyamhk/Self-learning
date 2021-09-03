const bcrypt = require('bcrypt');
const getCollection = require('../getCollection');

async function updateUser(email, updates = {}) {
  if (!email || Object.keys(updates).length === 0) {
    return {
      status: false,
      statusCode: 400,
      message: 'Missing data',
    };
  }

  try {
    const Users = await getCollection('Users');

    if (updates.username) {
      const isExisted = await Users.findOne({ username: updates.username });
      if (isExisted) {
        return {
          status: false,
          statusCode: 400,
          message: 'Username already exists',
        };
      }
    }

    const replacements = await genReplacement(updates);
    const result = await Users.updateOne({ email }, replacements, { upsert: false });

    if (result.modifiedCount === 1) {
      return {
        status: true,
        statusCode: 200,
        message: `User with email ${email} was updated successfully`,
      };
    }

    return {
      status: false,
      statusCode: 400,
      message: `User with email ${email} dose not exist`,
    };
  } catch (err) {
    throw err;
  }
};

async function genReplacement(fields) {
  try {
    const options = {};

    for (const key in fields) {
      const value = fields[key];

      if (value !== null && value !== undefined) {
        if (key === 'password') {
          const salt = await bcrypt.genSalt(10);
          const hash = await bcrypt.hash(fields.password, salt);
          options.password = hash;
        } else {
          options[key] = value;
        }
      }
    }

    options.updatedAt = new Date().toISOString();

    return { $set: options };
  } catch (err) {
    throw err;
  }
};

module.exports = updateUser;
