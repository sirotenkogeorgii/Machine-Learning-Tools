using Layers;
using Mathematics;

namespace Optimizers
{
    class SGD
    {
        private FullyConnectedLayer[] modelParameters;
        private float _learningRate;

        public SGD(FullyConnectedLayer[] parameters, float learningRate)
        {
            modelParameters = parameters;
            _learningRate = learningRate;
        }

        public void Update(int layerIndex, Matrix inputValues, Matrix previousGradient)
        {
            // Gradient for weights.
            Matrix weightsDerivatives = Matrix.EinSum(inputValues, previousGradient).ReduceMean(0);

            // Gradient for bias.
            Vector biasDerivatives = previousGradient.ReduceMean(0);

            // Update.
            modelParameters[layerIndex].Bias -= (_learningRate * biasDerivatives);
            modelParameters[layerIndex].Weights -= (_learningRate * weightsDerivatives);
        }
    }

    class Momentum
    {
        private FullyConnectedLayer[] modelParameters;
        private float _beta;
        private float _learningRate;
        private FullyConnectedLayer[] _momentum;

        public Momentum(FullyConnectedLayer[] parameters, float beta, float learningRate)
        {
            modelParameters = parameters;
            _learningRate = learningRate;
            _beta = beta;

            _momentum = new FullyConnectedLayer[modelParameters.Length];

            for (int i = 0; i < modelParameters.Length; i++) 
                _momentum[i] = new FullyConnectedLayer(modelParameters[i].Weights.Values.GetLength(0), modelParameters[i].Weights.Values.GetLength(1), "zeros");

        }

        public void Update(int layerIndex, Matrix inputValues, Matrix previousGradient)
        {
            // Momentum for weights.
            _momentum[layerIndex].Weights = _beta * _momentum[layerIndex].Weights - _learningRate * Matrix.EinSum(inputValues, previousGradient).ReduceMean(0);
            
            // Momentum for bias.
            _momentum[layerIndex].Bias = _beta * _momentum[layerIndex].Bias - _learningRate * previousGradient.ReduceMean(0);
            
            // Update.
            modelParameters[layerIndex].Bias += (_momentum[layerIndex].Bias);
            modelParameters[layerIndex].Weights += _momentum[layerIndex].Weights;
        }
    }
}