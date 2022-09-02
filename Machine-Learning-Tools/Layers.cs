using Models;
using System.Collections.Generic;
using Mathematics;
namespace Layers
{
    
    // Class of one fully connected neural network layer.
    class FullyConnectedLayer
    {
        // Layer weights.
        public Matrix Weights;

        // Bias for each layer neuron.
        public Vector Bias;

        // Learning rate.
        private float _lr;

        // Layers can be defined by the number of input values and the number of output values.
        public FullyConnectedLayer(int input, int output, float lr)
        {
            // We generate weights from a normal distribution
            // with given distribution parameters.
            Weights = Generate.RandomNormal(new int[] { input, output }, 0, 0.1f);

            // Initialize bias as zero vector.
            Bias = new Vector(output);

            _lr = lr;
        }

        // Backpropagation for a layer.
        public void BackProp(Matrix inputValues, Matrix previousGradient)
        {
            // Gradient for weights.
            Matrix weightsDerivatives = Matrix.EinSum(inputValues, previousGradient).ReduceMean(0);

            // Gradient for bias.
            Vector biasDerivatives = previousGradient.ReduceMean(0);

            // Update.
            Bias -= (_lr * biasDerivatives);
            Weights -= (_lr * weightsDerivatives);
        }
    }
    
    struct LayerSequence
    {
        public List<FullyConnectedLayer> FC;
        //public FullyConnectedLayer[] FC;
        //public List <ConvolutionalLayer> ConvLayer;
        public int Count;
        public int FcStartsIndex;

        public LayerSequence()
        {
            FC = new List<FullyConnectedLayer>();
            Count = 0;
            FcStartsIndex = 0;
        }
        public void Addo(FullyConnectedLayer fc)
        {
            FC.Add(fc);
            Count += 1;
        }

        // public ConvolutionalLayer GetConvLayer(int index)
        // {
        //     if (Count <= index)
        //         throw new Exception();
        //     // if (index < ConvLayer.Count)
        //     //     return ConvLayer[index];
        //     return FC[index];
        
        public FullyConnectedLayer GetFcLayer(int index)
        {
            if (FC.Count <= index)
                throw new Exception($"GetLayer: Index if out of range. Count of layers is {Count}, requested layer is {index}");
            return FC[index];
        }
    }
}