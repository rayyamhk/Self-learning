function optionsConstructor(obj) {
  const { order_by, order, limit, skip } = obj;
  const options = {};
  if (order_by && order) {
    const sort = {};
    sort[order_by] = order === 'ASC' ? 1 : -1;
    options.sort = sort;
  }
  if (limit !== undefined) {
    options.limit = parseInt(limit);
  }
  if (skip !== undefined) {
    options.skip = parseInt(skip);
  }
  return options;
};

module.exports = optionsConstructor;
