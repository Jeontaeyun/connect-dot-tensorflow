const tf = require("@tensorflow/tfjs-node");

const t1 = tf.scalar(1);
const t2 = tf.scalar(2);
const t3 = tf.scalar(5);
const t4 = tf.add(t1, t2);
const t5 = t4.div(t3);
tf.print(`t1-id: ${t1.id}`);
tf.print(`t2-id: ${t2.id}`);
tf.print(`t3-id: ${t3.id}`);
tf.print(`t4-id: ${t4.id}`);
tf.print(`t5-id: ${t5.id}`);
tf.print(t5);

const one = new Uint8Array(3);
one[0] = 100;
one[1] = 100;
const tensor1 = tf.tensor([1, 2, 3]);
const tensor2 = tf.tensor(one);
tf.print(tensor1);
tf.print(tensor2);

const list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
tf.print(tf.tensor(list, [2, 2, 3]));
tf.print(tf.tensor(list, [6, 2]));

const one1d = tf.tensor1d([1, 3, 5]);
const two1d = tf.tensor1d([2, 4, 6]);
const scalar1 = tf.scalar(4);
const result = tf.add(one1d, two1d);
const scalarResult = tf.add(one1d, scalar1);
tf.print(result);
tf.print(scalarResult);

const innerTarget1 = tf.tensor1d([1, 2, 3]);
const innerTarget2 = tf.tensor1d([4, 5, 6]);
const innerProductResult = tf.dot(innerTarget1, innerTarget2);
tf.print(innerProductResult);

const outerTarget1 = tf.tensor1d([1, 2, 3]);
const outerTarget2 = tf.tensor1d([3, 4, 5]);
const outerProductResult = tf.outerProduct(outerTarget1, outerTarget2);
tf.print(outerProductResult);

const matrix1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
const matrix2 = tf.tensor2d([5, 6, 7, 8], [2, 2]);
const matrix3 = tf.tensor2d([[3], [6]]);
// Matrix Element wise add
const matrixResult = tf.add(matrix1, matrix2);
tf.print(matrixResult);
// Matrix Element wise multiply
const matrixResult2 = tf.mul(matrix1, matrix3);
tf.print(matrixResult2);
//  Matrix Element wise division
const matrixResult3 = tf.div(matrix1, matrix3);
tf.print(matrixResult3);
//  Matrix dot product(Inner Product)
const matrixResult4 = tf.matMul(matrix1, matrix2);
tf.print(matrixResult4);

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
