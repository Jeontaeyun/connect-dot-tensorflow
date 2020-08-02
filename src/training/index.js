const tf = require("@tensorflow/tfjs-node");

module.exports = function runExport() {
  const batchSize = 25;
  // weight and bias is tf.tensor. create sample value with tf.randomNormal
  const weight = tf.variable(tf.randomNormal([1], 0, 1, "float32", 700));
  const bias = tf.variable(tf.randomNormal([1], 0, 1, "float32", 710));
  const learningRate = 0.05;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 0.1;
  // tf.train.adam is api for creating optimizer

  function predict(dataX) {
    return weight.mul(dataX).add(bias);
  }

  function loss(predict, dataY) {
    return predict.sub(dataY).square().mean();
  }

  const optimizer = tf.train.adam(learningRate, beta1, beta2, epsilon);

  const train = () => {
    // Machine Training
    for (let k = 0; k < 5; k++) {
      tf.print(`weight:${weight}\nbias:${bias}`);
      optimizer.minimize(() => {
        // calculate prediction Y
        const pred = predict(2);
        // calculate and return cost
        return loss(pred, 1);
      });
    }
  };

  train();
};
