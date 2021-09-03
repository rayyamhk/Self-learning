module.exports = {
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
};

// for any k-sequence
// generate all its (k-1)-subsequences
function getAllImmediateSubsequences(s) {
  function generate(sequence, size, index = 0, prevTransactionIdx = -1, subSequence = []) {

    if (getSquenceSize(subSequence) === size) {
      subSequences.push(clone(subSequence));
      return;
    }
    if (index === size + 1) {
      return;
    }

    let item, currentTransactionIdx, currentIdx = -1;
    for (let i = 0; i < sequence.length; i++) {
      const transaction = sequence[i];
      for (let j = 0; j < transaction.length; j++) {
        currentIdx++;
        if (currentIdx === index) {
          item = transaction[j];
          currentTransactionIdx = i;
          break;
        }
      }
      if (currentIdx === index) {
        break;
      }
    }

    if (currentTransactionIdx !== prevTransactionIdx) {
      subSequence.push([item]);
    } else {
      const lastT = subSequence.pop() || [];
      lastT.push(item);
      subSequence.push(lastT);
    }

    generate(sequence, size, index + 1, currentTransactionIdx, subSequence);

    if (currentTransactionIdx !== prevTransactionIdx) {
      subSequence.pop();
    } else {
      const lastT = subSequence.pop() || [];
      lastT.pop();
      subSequence.push(lastT);
    }

    generate(sequence, size, index + 1, prevTransactionIdx, subSequence);
  };

  const subSequences = [];
  const size = getSquenceSize(s) - 1;
  generate(s, size);

  const noDuplicated = [];
  for (let i = 0; i < subSequences.length; i++) {
    const t = subSequences[i];
    let isUnique = true;
    for (let j = 0; j < noDuplicated.length; j++) {
      if (isTransactionEqual(t, noDuplicated[j])) {
        isUnique = false;
        break;
      }
    }
    if (isUnique) {
      noDuplicated.push(t);
    }
  }

  return noDuplicated;
}

// given that s1 and s2 are from L, i.e. same size
// given that they can be merged
function mergeSequences(s1, s2) {
  const c1 = clone(s1);
  const lastTransaction2 = s2[s2.length - 1];
  const lastItem2 = lastTransaction2[lastTransaction2.length - 1];
  if (lastTransaction2.length === 1) {
    c1.push([lastItem2]);
  } else {
    c1[c1.length - 1].push(lastItem2);
  }
  return c1;
};

// given that s1 and s2 are from L, i.e. same size
function canMergeSequences(s1, s2) {
  const c1 = clone(s1), c2 = clone(s2);

  // remove first item from s1, in place
  const firstTransaction1 = c1[0], lastTransaction2 = c2[c2.length - 1];
  if (firstTransaction1.length === 1) {
    c1.shift();
  } else {
    c1[0].shift();
  }

  // remove last item from s2, in place
  if (lastTransaction2.length === 1) {
    c2.pop();
  } else {
    c2[c2.length - 1].pop();
  }

  return isSequenceEqual(c1, c2);
};

// clone a transaction or sequence
function clone(instance) {
  const copy = [];
  for (let i = 0; i < instance.length; i++) {
    if (Array.isArray(instance[i])) { // if instance is a sequence
      copy.push(clone(instance[i]));
    } else { // if instance is a transaction
      return [...instance];
    }
  }
  return copy;
};

function isSequenceEqual(s1, s2) {
  if (s1.length !== s2.length) {
    return false;
  }
  for (let i = 0; i < s1.length; i++) {
    const t1 = s1[i], t2 = s2[i];
    if (!isTransactionEqual(t1, t2)) {
      return false;
    }
  }
  return true;
};

function isTransactionEqual(t1, t2) {
  return isTransactionContain(t1, t2) && isTransactionContain(t2, t1);
};

function getSquenceSize(s) {
  let size = 0;
  for (let i = 0; i < s.length; i++) {
    const t = s[i];
    size += t.length;
  }
  return size;
};

function getSupportCount(dataset, s) {
  let counter = 0;
  for (let i = 0; i < dataset.length; i++) {
    const sequence = dataset[i];
    if (isSubsequence(sequence, s)) {
      counter++;
    }
  }
  return counter;
};

// returns true if s2 is subsequence of s1
// O(N^2)
function isSubsequence(s1, s2) {
  if (s1.length < s2.length) {
    return false;
  }
  let idx1 = 0, idx2 = 0;
  while (idx2 < s2.length && idx1 < s1.length) {
    const t1 = s1[idx1], t2 = s2[idx2];
    if (isTransactionContain(t1, t2)) {
      idx1++;
      idx2++;
    } else {
      idx1++;
    }
  }
  return idx2 === s2.length;
};

// returns true if t1 contains t2
// O(N)
function isTransactionContain(t1, t2) {
  if (t1.length < t2.length) {
    return false;
  }
  const hash = {};
  for (let i = 0; i < t1.length; i++) {
    const item = t1[i];
    hash[item] = true;
  }
  for (let i = 0; i < t2.length; i++) {
    const item = t2[i];
    if (!hash[item]) {
      return false
    }
  }
  return true;
};