function genVerification(len = 6) {
  const POOL = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
  ];
  const POOL_SIZE = 36;

  let code = '';
  for (let i = 0; i < len; i++) {
    const idx = Math.floor(Math.random() * POOL_SIZE);
    code += POOL[idx];
  }
  return code;
};

module.exports = genVerification;
