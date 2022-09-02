using System.Net;
using MNIST.IO;

namespace Datasets
{
    // MNIST dataset.
    public class Mnist
    {
        // Class for processing and forming a dataset.
        public class Dataset
        {
            // Batch generator.
            public IEnumerable<Batch> Batches(int batchSize)
            {
                if (batchSize > DataSize) batchSize = DataSize;

                int[] permutations = _shuffle ? Shuffle(DataSize) : Enumerable.Range(0, DataSize).ToArray();

                int reminder = DataSize;

                int batchCounter = batchSize;
                while (reminder > 0)
                {

                    // The last batch
                    if (reminder < batchSize)
                    {
                        batchCounter = DataSize;
                        batchSize = reminder;
                    }

                    float[][,] images = new float[batchSize][,];
                    float[] labels = new float[batchSize];

                    for (int i = batchCounter - batchSize; i < batchCounter; i++)
                    {
                        int index = i - (batchCounter - batchSize);

                        labels[index] = _labels[permutations[i]];
                        images[index] = _images[permutations[i]];
                    }
                    batchCounter += batchSize;
                    reminder -= batchSize;

                    yield return new Batch(images, labels);
                }
            }

            // Batch, that is, a part of a dataset of a certain size,
            // including an image-label pair.
            public struct Batch
            {
                public readonly float[][,] Images;
                public readonly float[] Labels;

                public Batch(float[][,] images, float[] labels)
                {
                    Images = images;
                    Labels = labels;
                }
            }
            // Whether to interfere with the data in the dataset.
            private readonly bool _shuffle;

            // Images in the dataset.
            private readonly float[][,] _images;

            // Target variables in the dataset.
            private readonly float[] _labels;

            // Number of images in the dataset.
            public readonly int DataSize;

            public Dataset(float[][,] images, float[] labels, string set, bool shuffle)
            {
                _shuffle = shuffle;

                // If the training dataset is being loaded,
                // then the number of images is 60000,
                // if the test one is 10000.
                DataSize = (set == "Train") ? 60000 : 10000;

                _images = images;
                _labels = labels;
            }

            // Shuffles an array with values from zero to dataSize.
            private int[] Shuffle(int dataSize)
            {
                var random = new Random();

                var arr = Enumerable.Range(0, dataSize);
                int[] shuffledArr = (from i in arr
                                     orderby random.Next()
                                     select i).ToArray();

                return shuffledArr;
            }
        };

        // Folder where the data will be downloaded.
        private readonly string _targetFolder = @"./data";

        // Links from where the data will be downloaded.
        private readonly string[] _links = {"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                                   "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                                   "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                                   "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"};

        // Train Dataset.
        public Dataset Train;

        // Test Dataset.
        public Dataset Test;

        // Downloading data from links.
        private void DownloadFileByURL(string[] links, string localDir)
        {
            // Create target folder if it doesn't exist.
            if (!Directory.Exists(localDir)) Directory.CreateDirectory(localDir);

            foreach (string link in links)
            {
                string file = link.Split('/')[^1];
                string localPath = Path.Combine(localDir, file);

                // If this file is not downloaded.
                if (!File.Exists(localPath))
                {
                    string log = $"Downloading file {file}...";
                    Console.WriteLine(log);
                    using (var client = new WebClient())
                        client.DownloadFile(link, localPath);
                }

            }
        }

        public Mnist()
        {
            DownloadFileByURL(_links, _targetFolder);

            // Dataset Creation.
            int i = 0;
            foreach (string set in new string[] { "Train", "Test" })
            {
                string[] files = new string[2];
                for (int j = 0; j < 2; j++)
                {
                    string file = _links[j + 2 * i].Split('/')[^1];
                    files[j] = Path.Combine(@"./data", file);

                }

                string log = $"{set} dataset preparation...";
                Console.WriteLine(log);
                float[][,] images = FileReaderMnist.LoadImages(files[0], set);
                float[] labels = FileReaderMnist.LoadLabel(files[1]);

                var setField = GetType().GetField(set);
                setField.SetValue(this, new Dataset(images, labels, set, set == "Train"));
                i++;
            }
        }
    }
}