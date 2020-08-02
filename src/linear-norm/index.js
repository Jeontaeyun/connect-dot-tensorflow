const tf = require("@tensorflow/tfjs-node");

module.exports = function runNorm() {
  const point1 = tf.tensor1d([1, 2, 1, 5]);
  const normResult = tf.norm(point1);
  tf.print(normResult);
};
