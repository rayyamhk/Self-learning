const dataset = require('./dataset');
const {
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
} = require('./utils');

const MIN_SUPPORT_COUNT = 3;

function Apriori(dataset, minSupportCount) {
  const items = getAllItems(dataset);
  let allFrequentItemsets = [];
  let L = getFrequentOneItemsets(items, minSupportCount);
  let C, k = 1;
  console.log('=================== L1 ===================');
  L.forEach((i) => console.log(i, getSupportCount(dataset, i)));

  allFrequentItemsets = [...L];
  while (L.length > 0) {
    k++;
    C = getCandidates(L);
    L = getFrequentItemsets(dataset, C, MIN_SUPPORT_COUNT);
    allFrequentItemsets = [...allFrequentItemsets, ...L];

    console.log(`=================== C${k} ===================`);
    C.forEach((i) => console.log(i, getSupportCount(dataset, i)));
    console.log(`=================== L${k} ===================`);
    L.forEach((i) => console.log(i, getSupportCount(dataset, i)));
  }
  console.log(`=================== Terminated ===================`);
  
  return allFrequentItemsets;
};

console.log(Apriori(dataset, MIN_SUPPORT_COUNT));
