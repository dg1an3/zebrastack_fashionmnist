using System;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Utils;
using NumSharp;
using System.Collections.Generic;
using Tensorflow.Keras;

namespace FashionMnistVaeTfNet
{
    class Program
    {
        static void Main(string[] args)
        {
            {
                var helloWorldArray = np.array(new int[,]
                {
                {4,3,7,7,0},
                {8,0,6,7,0}
                });

                var helloWorldTensor = tf.constant(helloWorldArray);
                Console.WriteLine(helloWorldTensor);

                var x = tf.Variable(10, name: "x");
                Console.Write(x);
                //using (var session = tf.Session())
                //{
                //    // session.run(x.initializer);
                //    var result = session.run(x);
                //}
            }

            {
                // var layers = new LayersApi();
                // input layer
                var inputs = keras.layers.Input(shape: (32, 32, 3), name: "img");
                // convolutional layer
                var x = keras.layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
                x = keras.layers.Conv2D(64, 3, activation: "relu").Apply(x);
                var block_1_output = keras.layers.MaxPooling2D(3).Apply(x);
                x = keras.layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_1_output);
                x = keras.layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
                var block_2_output = keras.layers.Add().Apply(new Tensorflow.Tensors(x, block_1_output));
                x = keras.layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_2_output);
                x = keras.layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
                var block_3_output = keras.layers.Add().Apply(new Tensorflow.Tensors(x, block_2_output));
                x = keras.layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
                x = keras.layers.GlobalAveragePooling2D().Apply(x);
                x = keras.layers.Dense(256, activation: "relu").Apply(x);
                x = keras.layers.Dropout(0.5f).Apply(x);
                // output layer
                var outputs = keras.layers.Dense(10).Apply(x);
                // build keras model
                var model = keras.Model(inputs, outputs, name: "toy_resnet");
                model.summary();
                // compile keras model in tensorflow static graph
                model.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
                    loss: keras.losses.CategoricalCrossentropy(from_logits: true),
                    metrics: new[] { "acc" });
                // prepare dataset
                var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();
                x_train = x_train / 255.0f;
                y_train = np_utils.to_categorical(y_train, 10);
                // training
                model.fit(x_train[new Slice(0, 2000)], y_train[new Slice(0, 2000)],
                          batch_size: 64,
                          epochs: 10,
                          validation_split: 0.2f);
            }

            {
                var sequential = keras.Sequential(
                    new List<ILayer>() 
                    {
                        keras.layers.Conv2D(32, 3, activation: "relu"),
                        keras.layers.Conv2D(64, 3, activation: "relu"),
                        keras.layers.MaxPooling2D(3),
                        keras.layers.Conv2D(64, 3, activation: "relu", padding: "same"),
                        keras.layers.Conv2D(64, 3, activation: "relu", padding: "same"),
                        keras.layers.Conv2D(64, 3, activation: "relu", padding: "same"),
                        keras.layers.Conv2D(64, 3, activation: "relu", padding: "same"),
                        keras.layers.Conv2D(64,3, activation: "relu", padding: "same"),
                    }, name: "myModel");
            }
        }
    }
}
