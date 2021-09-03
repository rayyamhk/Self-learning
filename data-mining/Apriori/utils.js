module.exports = {
  getAllItems,
  getFrequentOneItemsets,
  merge,
  getAllImmediateSubItemset,
  isEqual,
  isIncluded,
  canMerge,
  isContain,
  getSupportCount,
  getCandidates,
  getFrequentItemsets
};

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

function getFrequentOneItemsets(items, minSupportCount) {
  const frequentItems = [];
  Object.entries(items).forEach(([key, value]) => {
    if (value >= minSupportCount) {
      frequentItems.push([key]);
    }
  });
  return frequentItems;
};

function merge(i1, i2) {
  const cache = {};
  for (let i = 0; i < i1.length; i++) {
    cache[i1[i]] = true;
  }
  for (let i = 0; i < i2.length; i++) {
    cache[i2[i]] = true;
  }
  return Object.keys(cache);
};

function isEqual(i1, i2) {
  if (i1.length !== i2.length) {
    return false;
  }
  const cache1 = {}, cache2 = {};
  for (let i = 0; i < i1.length; i++) {
    cache1[i1[i]] = true;
  }
  for (let i = 0; i < i2.length; i++) {
    if (!cache1[i2[i]]) {
      return false;
    }
    cache2[i2[i]] = true;
  }
  for (let i = 0; i < i1.length; i++) {
    if (!cache2[i1[i]]) {
      return false;
    }
  }
  return true;
};

function isIncluded(itemsets, itemset) {
  for (let i = 0; i < itemsets.length; i++) {
    if (isEqual(itemsets[i], itemset)) {
      return true;
    }
  }
  return false;
};

function canMerge(i1, i2, frequentItemsets) {
  const cache = {};
  for (let i = 0; i < i1.length; i++) {
    const item = i1[i];
    cache[item] = true;
  }

  let k = 0;
  for (let i = 0; i < i2.length; i++) {
    const item = i2[i];
    if (cache[item]) {
      k++;
    }
  }

  if (k !== i1.length - 1) {
    return false;
  }

  const merged = merge(i1, i2);
  const subItemsets = getAllImmediateSubItemset(merged);
  for (let i = 0; i < subItemsets.length; i++) {
    if (!isIncluded(frequentItemsets, subItemsets[i])) {
      return false;
    }
  }
  return true;
};

function isContain(i1, i2) {
  const cache = {};
  for (let i = 0; i < i1.length; i++) {
    cache[i1[i]] = true;
  }
  for (let i = 0; i < i2.length; i++) {
    if (!cache[i2[i]]) {
      return false;
    }
  }
  return true;
}

function getAllImmediateSubItemset(i) {
  function generate(itemset, size, index = 0, subItemset = []) {
    if (subItemset.length === size) {
      subItemsets.push([...subItemset]);
      return;
    }
    if (index === size + 1) {
      return;
    }
    const item = itemset[index];
    generate(itemset, size, index + 1, [...subItemset, item]);
    generate(itemset, size, index + 1, [...subItemset]);
  };

  const subItemsets = [];
  const size = i.length - 1;
  generate(i, size);

  return subItemsets;
};

function getSupportCount(dataset, itemset) {
  let count = 0;
  dataset.forEach((i) => {
    if (isContain(i, itemset)) {
      count++;
    }
  })
  return count;
};

function getCandidates(frequentItemsets) {
  const candidates = [];
  for (let i = 0; i < frequentItemsets.length; i++) {
    for (let j = i + 1; j < frequentItemsets.length; j++) {
      const itemset1 = frequentItemsets[i];
      const itemset2 = frequentItemsets[j];
      if (canMerge(itemset1, itemset2, frequentItemsets)) {
        candidates.push(merge(itemset1, itemset2));
      }
    }
  }
  const uniqueCandidates = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    let isUnique = true;
    for (let j = 0; j < uniqueCandidates.length; j++) {
      if (isEqual(c, uniqueCandidates[j])) {
        isUnique = false;
      }
    }
    if (isUnique) {
      uniqueCandidates.push(c);
    }
  }
  return uniqueCandidates;
};

function getFrequentItemsets(dataset, candidates, minSupportCount) {
  const frequentItemsets = [];
  candidates.forEach((itemset) => {
    const sc = getSupportCount(dataset, itemset);
    if (sc >= minSupportCount) {
      frequentItemsets.push(itemset);
    }
  });
  return frequentItemsets;
};
