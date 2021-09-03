const logger = require('./logger');

class Cache {
  /**
   * ttl: time-to-live in ms, 0 means never expire
   * period: time interval of scheduler in ms, 0 means no scheduler
   */
  constructor(options = {}) {
    const { ttl = 0, period = 0 } = options;
    if (typeof ttl !== 'number' || ttl < 0 || typeof ttl !== 'number' || period < 0) {
      throw new Error('Invalid arguments');
    }
    this.cache = {};
    this.ttl = ttl; // in ms
    this.period = period; // in ms
    this.size = 0;
    this.timer = null;
    this.isRunning = false;
  };

  set(key, value) {
    if (this.cache[key] === undefined) {
      this.size++;
    }
    if (this.ttl > 0) {
      this.cache[key] = {
        value,
        expiredAt: Date.now() + this.ttl,
      };
    } else {
      this.cache[key] = value;
    }
    this._setTimer();
  };

  remove(key) {
    if (this.cache[key] !== undefined) {
      this.size--;
    }
    delete this.cache[key];
    this._setTimer();
  };

  get(key) {
    if (this.cache[key] === undefined || key === undefined || key === null) {
      return undefined;
    }

    if (this.ttl > 0) {
      const { value, expiredAt } = this.cache[key];
      if (expiredAt > Date.now()) {
        return value;
      }
      this.remove(key);
      return undefined;
    }

    // never expired
    return this.cache[key];
  };

  /**
   * Flush all expired items
   */
  flush() {
    if (this.ttl > 0) {
      const now = Date.now();
      for (const key in this.cache) {
        const { expiredAt } = this.cache[key];
        if (expiredAt <= now) {
          this.remove(key);
        }
      }
      return;
    }
    this.cache = {};
  };

  _setTimer() {
    if (this.ttl === 0 || this.period === 0) {
      return;
    }

    if (this.size > 0 && !this.isRunning) {
      logger.info(`${new Date().toLocaleTimeString()}: Schedule starts...`);

      this.timer = setInterval(() => {
        this.flush();
        logger.info(`${new Date().toLocaleTimeString()}: Flushed, ${this.size} items still alive.`);
      }, this.period);

      this.isRunning = true;
      return;
    }

    if (this.size === 0 && this.isRunning) {
      logger.info(`${new Date().toLocaleTimeString()}: Schedule ends...`);
      clearInterval(this.timer);
      this.isRunning = false;
    }
  };
}

module.exports = Cache;
