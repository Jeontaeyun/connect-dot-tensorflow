const tf = require("@tensorflow/tfjs-node");

module.exports = function runMSE() {
  // 01. Initial data set
  const dataX = [1, 2, 3, 4, 5];
  const dataY = [10, 20, 30, 40, 50];
  const trainDataX = tf.tensor1d(dataX);
  const trainDataY = tf.tensor1d(dataY);

  const weight = tf.variable(tf.randomNormal([1], 0, 1, "float32", 700));
  const bias = tf.variable(tf.randomNormal([1], 0, 1, "float32", 710));

  // 02. Create optimizer
  const learningRate = 0.05;
  const optimizer = tf.train.sgd(learningRate);

  // 03. Declare prediction function
  const predict = (_dataX) => {
    return weight.mul(_dataX).add(bias);
  };

  const loss = (predict, _dataY) => {
    // 04. square() is element wise but mean() is not element wise
    return predict.sub(_dataY).square().mean();
  };
  // 05. Learning with optimizer
  for (let k = 0; k < 200; k++) {
    tf.print(`[weight]${weight}\n[bias]${bias}`);
    optimizer.minimize(() => {
      return loss(predict(trainDataX), trainDataY);
    });
  }
};
