const tf = require("@tensorflow/tfjs-node");

module.exports = function runDeviation() {
  const value = tf.tensor1d([1, 3, 5, 7, 9]);
  const mean = value.mean();
  tf.print(mean);

  const deviation = value.sub(mean);
  tf.print(deviation);

  const square = deviation.square();
  tf.print(square);

  const variance = square.mean();
  tf.print(variance);

  const standard = variance.sqrt();
  tf.print(standard);
};
