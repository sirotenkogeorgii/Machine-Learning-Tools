using Datasets;
using MatthiWare.CommandLine;
using Mathematics;
using Models;
using MatthiWare.CommandLine.Core.Models;

public class Program
{
    static void Main(string[] args)
    {
        // Command line parser options:
        // "--" before parameter name,
        // "=" after parameter name.
        var options = new CommandLineParserOptions
        {
            PrefixLongOption = "--",
            PostfixOption = "=",
        };

        var parser = new CommandLineParser<Options.ModelOptions>(options);
        var argumentParsing = parser.Parse(args);

        if (argumentParsing.HasErrors)
        {
            Console.Error.WriteLine("Wrong parsing!");
            return;
        }

        // Parsed args.
        var arguments = argumentParsing.Result;

        var dataset = new Mnist();
        var model = new MnistModel(arguments);


        for (int epoch = 0; epoch < arguments.Epochs; epoch++)
        {
            model.TrainEpoch(dataset.Train);
            float[] accCorrects = model.Evaluate(dataset.Test);
            Console.WriteLine($"Epoch: {epoch + 1}, Accuracy: {accCorrects[0]}, Corrects: {accCorrects[1]}");
        }
    }
}