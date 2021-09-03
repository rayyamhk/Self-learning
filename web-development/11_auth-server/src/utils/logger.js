const chalk = require('chalk');

module.exports = {
  log: (args) => console.log(`${new Date().toLocaleTimeString()}: ${chalk.green(args)}`),
  info: (args) => console.info(`${new Date().toLocaleTimeString()}: ${chalk.yellowBright(args)}`),
  error: (args) => console.error(`${new Date().toLocaleTimeString()}: ${chalk.redBright(args)}`),
  warn: (args) => console.warn(`${new Date().toLocaleTimeString()}: ${chalk.keyword('orange')(args)}`),
};
