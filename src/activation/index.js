const tf = require("@tensorflow/tfjs-node");

const data = tf.linspace(-5, 5, 100);
tf.sigmoid(data);

const model = tf.sequential();

model.add(tf.layers.dense({ activation: "relu" }));
model.add(tf.layers.dense({ activation: "sigmoid" }));
