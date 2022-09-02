using MatthiWare.CommandLine.Core.Attributes;

namespace Options
{

    // Setting model hyperparameters.
    public class ModelOptions
    {

        [Name("batch_size", "batchsize"), Description("Batch size."), DefaultValue(64)]
        public int BatchSize { get; set; }

        [Name("epochs", "epochs"), Description("Number of epochs."), DefaultValue(2)]
        public int Epochs { get; set; }

        [Name("learning_rate", "learning_rate"), Description("Learning rate."), DefaultValue(0.1f)]
        public float LearningRate { get; set; }

        // 784 (=28*28) means input vector and 10 means output vector,
        // which corresponds to the number of classes.
        [Name("architecture", "architecture"), Description("NN Architecture."), DefaultValue("784-40-20-10")]
        public string Architecture { get; set; }

    }
}