const tf = require("@tensorflow/tfjs-node");
const { tensor } = require("@tensorflow/tfjs-node");

module.exports = function () {
  // iris [length, width]
  const irisX = [
    [1.5, 0.2],
    [1.5, 0.4],
    [4.7, 1.4],
    [5, 1.7],
    [5.1, 1.9],
    [4.5, 1.7],
  ];
  const irisY = [0, 0, 1, 1, 2, 2];
  const trainX = tf.tensor2d(irisX);
  // dataY = [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]
  const oneHotY = tf.oneHot(irisY, 3);
  const trainY = tf.cast(oneHotY, "float32");

  const weight = tf.variable(
    tf.randomNormal([2, 3], 0, 1, "float32", 701),
    true,
    "weight"
  );

  const bias = tf.variable(
    tf.randomNormal([3], 0, 1, "float32", 701),
    true,
    "bias"
  );

  const learningRate = 0.3;
  const optimizer = tf.train.adam(learningRate);
  // matMul is dot-product
  const loss = () => {
    const logit = trainX.matMul(weight).add(bias);
    return tf.losses.softmaxCrossEntropy(trainY, logit);
  };

  const train = () => {
    for (let k = 0; k < 1000; k++) {
      optimizer.minimize(() => loss());
    }
  };

  train();

  const predict = (dataX) => dataX.matMul(weight).add(bias).softmax();
  const predictData = tf.tensor2d([
    [4, 1.2],
    [5, 2],
  ]);

  const result = tf.unstack(predict(predictData)).map((pred) => {
    // argMax is return max index in vector
    const maxIndex = tf.argMax(pred).dataSync();
    return maxIndex;
  });

  tf.print(result);
};
