using System;
using static Tensorflow.Binding;
using NumSharp;

namespace FashionMnistVaeTfNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var helloWorldArray = np.array(new int[,]
            {
                {4,3,7,7,0},
                {8,0,6,7,0}
            });

            var helloWorldTensor = tf.constant(helloWorldArray);
            Console.WriteLine(helloWorldTensor);
        }
    }
}
