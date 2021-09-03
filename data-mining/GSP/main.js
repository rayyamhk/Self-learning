const dataset = require('./dataset');
const {
  getAllImmediateSubsequences,
  mergeSequences,
  canMergeSequences,
  clone,
  isSequenceEqual,
  isTransactionEqual,
  getSquenceSize,
  getSupportCount,
  isSubsequence,
  isTransactionContain
} = require('./utils');

function GSP(dataset, support = 0.01) {

  const N = dataset.length;
  let allFrequents = [];

  const cache = {};
  dataset.forEach((data) => {
    data.forEach((transaction) => {
      transaction.forEach((item) => {
        cache[item] = true;
      });
    });
  });
  let candidates = Object.keys(cache).sort().map((item) => [ [item] ]);
  let frequents = candidates.filter((s) => getSupportCount(dataset, s) >= N * support);
  
  console.log('Pass: 1');
  console.log('==================================Candidates==================================');
  for (let i = 0; i < candidates.length; i++) {
    const s = candidates[i];
    const sc = getSupportCount(dataset, s);
    console.log(s, sc);
  }
  console.log('==================================Frequents==================================');
  console.log(frequents);
  console.log('========================================================================');

  let size = 2;
  while (frequents.length !== 0) {
    candidates = getCandidates(frequents); // Candidates generation

    console.log('Pass: ', size);
    console.log('==================================Candidates==================================');
    for (let i = 0; i < candidates.length; i++) {
      const s = candidates[i];
      const sc = getSupportCount(dataset, s);
      console.log(s, sc);
    }

    candidates = pruneCandidates(candidates, frequents);
    console.log('==================================Pruned Candidates==================================');
    console.log(candidates);

    frequents = candidates.filter((s) => getSupportCount(dataset, s) >= N * support);
    console.log('==================================Frequents==================================');
    console.log('Frequents: ', frequents);
    console.log('========================================================================');
    
    allFrequents = [...allFrequents, ...frequents];

    size++;
  }
  return allFrequents;
};

function pruneCandidates(C, prevL) {
  const goodCandidates = [];

  for (let i = 0; i < C.length; i++) {
    const s = C[i];
    const subSeqs = getAllImmediateSubsequences(s);
    let isGoodCandidate = true;

    for (let j = 0; j < subSeqs.length; j++) {
      const subS = subSeqs[j];
      let isContain = false;

      for (let k = 0; k < prevL.length; k++) {
        if (isSequenceEqual(subS, prevL[k])) {
          isContain = true;
          break;
        }
      }

      if (!isContain) {
        isGoodCandidate = false;
        break;
      }
    }

    if (!isGoodCandidate) {
      continue;
    }
    goodCandidates.push(s);
  }
  return goodCandidates;
};

// given that L is any valid frequence k-sequence
// i.e. all sequences have same size k
function getCandidates(L) {
  if (L.length === 0) {
    return [];
  }

  const size = getSquenceSize(L[0]);
  const C = [];

  if (size === 1) { // special case
    for (let i = 0; i < L.length; i++) {
      for (let j = i; j < L.length; j++) {
        const item1 = L[i][0][0], item2 = L[j][0][0];
        if (j === i) {
          C.push([
            [item1], [item1]
          ]);
        } else {
          C.push([
            [item1], [item2]
          ]);
          C.push([
            [item1, item2]
          ]);
          C.push([
            [item2], [item1]
          ]);
        }
      }
    }
    return C;
  }

  for (let i = 0; i < L.length; i++) {
    for (let j = 0; j < L.length; j++) {
      const s1 = L[i], s2 = L[j];
      if (canMergeSequences(s1, s2)) {
        C.push(mergeSequences(s1, s2));
      }
    }
  }
  return C;
};

const result = GSP(dataset, 0.65);
console.log(result);