const requestIp = require('request-ip');

function ipService() {
  return (req, res, next) => {
    req.requestIp = requestIp.getClientIp(req);
    next();
  };
};

module.exports = ipService;
