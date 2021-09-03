const getCollection = require('./getCollection');
const logger = require('./logger');
const optionsConstructor = require('./optionsConstructor');
const Cache = require('./Cache');
const genVerification = require('./genVerification');
const sendMail = require('./sendMail');

module.exports = {
  optionsConstructor,
  getCollection,
  genVerification,
  sendMail,
  logger,
  Cache,
};
