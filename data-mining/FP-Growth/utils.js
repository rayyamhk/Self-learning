const printTree = require('print-tree');

function getAllItems(dataset) {
  const items = {};
  dataset.forEach((transaction) => {
    transaction.forEach((item) => {
      if (items[item]) {
        items[item]++;
      } else {
        items[item] = 1;
      }
    });
  });
  return items;
};

function getFrequentItems(items, minSup) {
  const frequentItems = {}
  Object.entries(items).forEach(([key, value]) => {
    if (value >= minSup) {
      frequentItems[key] = value;
    }
  });
  return frequentItems;
};

function getSortedFrequentItems(frequentItems) {
  const temp = Object.entries(frequentItems);
  const sorted = {};
  temp.sort((a, b) => {
    return b[1] - a[1];
  });
  temp.forEach(([key, value]) => {
    sorted[key] = value;
  });

  return sorted;
};

function getFrequentItemsBought(transaction, sortedFrequentItemsKey) {
  const filtered = transaction.filter((item) => sortedFrequentItemsKey.includes(item));
  return filtered.sort((a, b) => sortedFrequentItemsKey.indexOf(a) - sortedFrequentItemsKey.indexOf(b));
};

class Tree {
  constructor(key, value) {
    this.key = key || 'ROOT';
    this.value = value || -1;
    this.childKeys = [];
    this.childRefs = [];
  }

  insert(transaction) {
    let self = this;
    for (let i = 0; i < transaction.length; i++) {
      const item = transaction[i];
      const idx = self.childKeys.indexOf(item);
      if (idx === -1) {
        const node = new Tree(item, 1);
        if (i + 1< transaction.length) {
          node.insert(transaction.slice(i + 1));
        }
        self.childKeys.push(item);
        self.childRefs.push(node);
        break;
      } else {
        self = self.childRefs[idx];
        self.value++;
      }
    }
  }
}

function buildTree(dataset, sortedFrequentItemsKey) {
  const FPTree = new Tree();
  dataset.forEach((transaction) => {
    const frequentItemsBought = getFrequentItemsBought(transaction, sortedFrequentItemsKey);
    FPTree.insert(frequentItemsBought);
  });
  const fp = JSON.parse(JSON.stringify(FPTree));
  printTree(
    fp,
    node => `${node.key} ${node.value}`,
    node => node.childRefs
  );  
};

module.exports = {
  getAllItems,
  getFrequentItems,
  getSortedFrequentItems,
  getFrequentItemsBought,
  buildTree,
  Tree,
};