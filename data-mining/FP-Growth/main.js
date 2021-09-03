const dataset = require('./dataset');
const {
  getAllItems,
  getFrequentItems,
  buildTree,
} = require('./utils');

const MIN_SUPPORT = 3;

const items = getAllItems(dataset);

const frequentItems = getFrequentItems(items, MIN_SUPPORT);
// console.log(frequentItems);

// do it yourself!
const sortedFrequentItemsKey = ['f', 'c', 'a', 'b', 'm', 'p'];

buildTree(dataset, sortedFrequentItemsKey);
