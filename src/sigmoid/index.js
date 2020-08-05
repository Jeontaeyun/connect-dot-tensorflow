const tf = require("@tensorflow/tfjs-node");

module.exports = function runNorm() {
  // create 100 data-set between -5 and 5 with same interval
  const dataX = tf.linspace(-5, 5, 100);
  // return sigmoid function value
  const dataY = tf.sigmoid(dataX);

  const weight = tf.variable(
    tf.randomNormal([1], 0, 1, "float32", 700),
    true,
    "weight"
  );
  const bias = tf.variable(
    tf.randomNormal([1], 0, 1, "float32", 710),
    true,
    "bias"
  );

  const loss = (trainX, trainY) => {
    const logit = trainX.mul(weight).add(bias);
    return tf.losses.sigmoidCrossEntropy(trainY, logit);
  };

  const learningRate = 0.2;
  const optimizer = tf.train.adam(learningRate);

  const train = () => {
    for (let k = 0; k < 500; k++) {
      optimizer.minimize(() => {
        return loss(dataX, dataY);
      });
    }
  };

  train();

  const predict = (X) => {
    const logit = X.mul(weight).add(bias);
    const values = logit.sigmoid();
    return values.round().toFloat();
  };

  const accuracy = predict(dataX);

  tf.print(`expected 1\nresult ${predict(tf.tensor(1))}`);
  tf.print(`expected 0\nresult ${predict(tf.tensor(-1))}`);
  tf.print(`expected 1\nresult ${predict(tf.tensor(1.7))}`);
  tf.print(`expected 0\nresult ${predict(tf.tensor(-0.2))}`);
};
