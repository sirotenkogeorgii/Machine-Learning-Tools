using Mathematics;
using Datasets;

namespace Models
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
    class MnistModel
    {
        // Model hyperparameters.
        private readonly Options.ModelOptions _Arguments;

        // An array of model layers.
        private FullyConnectedLayer[] _Layers;

        public MnistModel(Options.ModelOptions arguments)
        {
            _Arguments = arguments;

            string[] layers = _Arguments.Architecture.Split('-');

            if (layers.Length <= 2) layers = new string[3] { "784", "20", "10"};
            if (layers[^1] != "10") layers[^1] = "10";
            if (layers[0] != "784") layers[0] = "784";

            _Layers = new FullyConnectedLayer[layers.Length - 1];

            int result;
            // Create network words and configure them to match the desired architecture.
            for (int i = 0; i < _Layers.Length; i++)
            {
                var layerIsNum = int.TryParse(layers[i], out result);
                var nextLayerIsNum = int.TryParse(layers[i + 1], out result);
                if (layerIsNum && nextLayerIsNum)
                    _Layers[i] = new FullyConnectedLayer(Int32.Parse(layers[i]), Int32.Parse(layers[i + 1]), _Arguments.LearningRate);

            }
        }

        // Batch prediction.
        public Tensor Predict(float[][,] batchImages)
        {
            // Create an output tensor.
            var outputs = new Tensor(_Layers.Length + 1);

            // Wrap the batch images into tensor.
            var images = new Tensor(batchImages);

            // Flat batch.
            Matrix lastLayer = images.Tensor2Matrix(batchImages.Length);

            // The second matrix of the output tensor is flatten batch.
            outputs.Values[1] = lastLayer.Values;

            // Pass images through all layers.
            for (int i = 0; i < _Layers.Length - 1; i++)
            {
                lastLayer = ((lastLayer & _Layers[i].Weights) + _Layers[i].Bias).Tanh();
                outputs.Values[i + 2] = lastLayer.Values;
            }

            // We don't use the activation function for the last layer,
            // but we use softmax fucntion to get probability distribution.
            lastLayer = ((lastLayer & _Layers[_Layers.Length - 1].Weights) + _Layers[_Layers.Length - 1].Bias);

            // The probability distribution is the first matrix in the output tensor.
            outputs.Values[0] = lastLayer.Softmax().Values;

            return outputs;
        }

        // Training one epoch of the neural network,
        // i.e. going through all the images of the training dataset
        // and making an update based on each batch.
        public void TrainEpoch(Mnist.Dataset dataset)
        {
            // We go through each batch once per epoch.
            foreach (var batch in dataset.Batches(_Arguments.BatchSize))
            {
                // Make a batch prediction.
                Tensor predictions = Predict(batch.Images);

                // Onehot encoding for target variables.
                var goldLabels = new Vector(batch.Labels).OneHot(10);

                // The output layer of the neural network.
                var outputLayer = new Matrix(predictions.Values[0]);

                // The previous gradient at the very beginning is the gradient of the loss function
                // over the output values of the last layer before softmax was applied.
                Matrix prevGrad = outputLayer - goldLabels;

                // The gradient of the loss function over the output layer before applying the activation function.
                Matrix dCE_dbeforeTanh;

                // Derivative of the hyperbolic tangent with respect to the argument.
                Matrix TanhJDeriv;

                // The hidden layer after applying the activation function.
                Matrix hiddenAfterTanh;

                // We compute the gradients for each layer
                // starting from the end, that is, from the layer that is closest to the output.
                for (int i = _Layers.Length; i > 0; i--)
                {
                    hiddenAfterTanh = new Matrix(predictions.Values[i]);
                    TanhJDeriv = 1 - (hiddenAfterTanh * hiddenAfterTanh);
                    dCE_dbeforeTanh = (prevGrad & _Layers[i - 1].Weights.Transpose()) * TanhJDeriv;

                    _Layers[i - 1].BackProp(hiddenAfterTanh, prevGrad);

                    prevGrad = dCE_dbeforeTanh;
                }
            }
        }

        // Evaluate the Model on the Test Dataset.
        public float[] Evaluate(Mnist.Dataset dataset)
        {

            float correct = 0;
            foreach (var batch in dataset.Batches(_Arguments.BatchSize))
            {
                // Make a batch prediction.
                Tensor predictions = Predict(batch.Images);

                // The output layer of the neural network.
                var outputLayer = new Matrix(predictions.Values[0]);

                // Labels of the batch images.
                float[] batchLabels = batch.Labels;

                //  The most probabilistic classes are the model prediction.
                float[] maxProbs = outputLayer.ArgMax(1).Values;

                for (int observ = 0; observ < maxProbs.Length; observ++)
                    if (maxProbs[observ] == batchLabels[observ]) correct++;
            }

            return new float[] { correct / dataset.DataSize, correct };
        }
    }
}
