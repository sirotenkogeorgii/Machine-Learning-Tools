using MatthiWare.CommandLine;
using Mathematics;

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
    }
}