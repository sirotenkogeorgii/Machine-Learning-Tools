using System.IO.Compression;

namespace MNIST.IO
{
    public static class FileReaderMnist
    {
        public static float[][,] LoadImages(string imageFile, string set)
        {
            int dataSize = (set == "Train") ? 60000 : 10000;

            float[][,] images = new float[dataSize][,];

            using (var raw = File.OpenRead(imageFile))
            {
                using (var gz = new GZipStream(raw, CompressionMode.Decompress))
                {
                    using (var reader = new BinaryReaderMSB(gz))
                    {
                        var header = reader.ReadInt32MostSignByte();
                        if (header != 0x803) throw new InvalidDataException(header.ToString("x"));
                        var itemCount = reader.ReadInt32MostSignByte();
                        var rowCount = reader.ReadInt32MostSignByte();
                        var colCount = reader.ReadInt32MostSignByte();

                        for (var i = 0; i < itemCount; i++)
                        {
                            var image = new float[rowCount, colCount];

                            for (var r = 0; r < rowCount; r++)
                            {
                                for (var c = 0; c < colCount; c++)
                                {
                                    float value = (float)(reader.ReadByte() / 255.0);
                                    image[r, c] = value;
                                }
                            }
                            images[i] = image;
                        }
                    }
                }
            }

            return images;
        }

        public static float[] LoadLabel(string labelFile)
        {
            using (var raw = File.OpenRead(labelFile))
            {
                using (var gz = new GZipStream(raw, CompressionMode.Decompress))
                {
                    using (var reader = new BinaryReaderMSB(gz))
                    {
                        // Check Header / Magic Number
                        var header = reader.ReadInt32MostSignByte();
                        if (header != 0x801) throw new InvalidDataException(header.ToString("x"));
                        var itemCount = reader.ReadInt32MostSignByte();

                        return Array.ConvertAll(reader.ReadBytes(itemCount), item => (float)item);
                    }
                }
            }
        }
    }
}