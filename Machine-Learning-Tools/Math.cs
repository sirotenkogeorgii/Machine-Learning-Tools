namespace Mathematics
{

    // Generates various vectors, matrices and tensors.
    public class Generate
    {
        // Generates a matrix of the given size
        // where all elements are from a normal distribution
        // with the given mean and standard deviation.
        public static Matrix RandomNormal(int[] size, float mean = 0, float stddev = 1)
        {
            if (size.Length != 2)
                throw new Exception("RandomNormal Matrix: Wrong matrix shape. The matrix must have dimensions of length two. For example: { 2, 4 }.");

            var rand = new Random();

            var normMatrix = new Matrix(size);

            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    float randNormal = mean + stddev * (float)randStdNormal;
                    normMatrix.Values[i, j] = randNormal;
                }
            }
            return normMatrix;
        }

        // Generates a vactor of the given size
        // where all elements are from a normal distribution
        // with the given mean and standard deviation.
        public static Vector RandomNormal(int size, float mean = 0, float stddev = 1)
        {
            var rand = new Random();

            var normArray = new Vector(size);

            for (int i = 0; i < size; i++)
            {
                double u1 = 1.0 - rand.NextDouble();
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                float randNormal = mean + stddev * (float)(randStdNormal);
                normArray.Values[i] = randNormal;
            }
            return normArray;
        }

        // Generates a matrix of the given size
        // where all elements are random.
        public static Matrix RandomMatrix(int[] size)
        {
            if (size.Length != 2)
                throw new Exception("RandomMatrix: Wrong matrix shape. The matrix must have dimensions of length two. For example: { 2, 4 }.");

            var random = new Random();

            var zeroArray = new Matrix(size);

            for (int i = 0; i < size[0]; i++)
                for (int j = 0; j < size[1]; j++)
                    zeroArray.Values[i, j] = random.Next();

            return zeroArray;
        }

        // Generates a vector of the given size
        // where all elements are random.
        public static Vector RandomVector(int size)
        {
            var random = new Random();
            var zeroArray = new Vector(size);

            for (int i = 0; i < size; i++)
                zeroArray.Values[i] = random.Next();

            return zeroArray;
        }

        // Generates a tensor of the given size
        // where all elements are random.
        public static Tensor RandomTensor(int[] size)
        {
            if (size.Length != 3)
                throw new Exception("RandomTensor: Wrong tensor shape. The tensor must have dimensions of length three. For example: { 2, 4, 6 }.");

            var random = new Random();

            var zeroArray = new Tensor(new int[] { size[0], size[1], size[2] });

            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                    for (int k = 0; k < size[2]; k++)
                        zeroArray.Values[i][j, k] = random.Next();
            }

            return zeroArray;
        }

    }

    // Represents a vector and related operations.
    public class Vector
    {
        // Vector values.
        public float[] Values;

        // Thanks to this constructor, we can explicitly define a vector with specific values.
        public Vector(float[] inputValues)
        {
            if (inputValues.Length == 0)
                throw new Exception("Vector: Wrong vector size. Vector size cannot be 0.");

            Values = new float[inputValues.Length];
            for (int i = 0; i < inputValues.Length; i++) Values[i] = inputValues[i];
        }

        // Thanks to this constructor we can create a vector of a certain size with all zeros.
        public Vector(int size)
        {
            if (size < 1)
                throw new Exception("Vector: Wrong vector size. Vector size cannot be less than 1.");

            Values = new float[size];
        }

        // Returns the length of the vector.
        public int Shape()
        { 
            return Values.Length; 
        }

        // Prints the shape of the vector.
        public void PrintShape()
        {
            Console.WriteLine("[{0}, ]", Values.Length);
        }

        // Prints vector.
        public void Print()
        {
            foreach (float element in Values) Console.Write("{0} ", element);
            Console.WriteLine();
        }

        // Transposes the vector,
        // but since the vector in this program is represented as an array of numbers,
        // the output of the method will be a matrix with one column.
        public Matrix Transpose()
        {
            var transposedVector = new Matrix(new int[] { Values.Length, 1 });
            for (int i = 0; i < Values.Length; i++) transposedVector.Values[i, 0] = Values[i];
            return transposedVector;
        }

        // Multiplying a vector by a scalar multiplies each element of the vector by a scalar.
        public static Vector operator *(Vector A, float scalar)
        {
            var resultVector = new Vector(A.Shape());
            for (int i = 0; i < A.Shape(); i++) resultVector.Values[i] = A.Values[i] * scalar;
            return resultVector;
        }

        // Multiplying a vector by a scalar multiplies each element of the vector by a scalar.
        public static Vector operator *(float scalar, Vector A)
        {
            return A * scalar;
        }

        // Element-wise multiplication of two vectors, not dot product.
        public static Vector operator *(Vector A, Vector B)
        {
            if (A.Shape() != B.Shape())
                throw new Exception($"Vector Element-wise Multiplication: Incorrect sizes: {A.Shape()} and {B.Shape()}.");

            var resultVector = new Vector(A.Shape());

            for (int i = 0; i < A.Shape(); i++) resultVector.Values[i] = A.Values[i] * B.Values[i];

            return resultVector;
        }

        // Add a scalar to each element of the vector.
        public static Vector operator +(Vector A, float scalar)
        {
            var resultVector = new Vector(A.Shape());
            for (int i = 0; i < A.Shape(); i++)
                resultVector.Values[i] = A.Values[i] + scalar;
            return resultVector;
        }

        // Add a scalar to each element of the vector.
        public static Vector operator +(float scalar, Vector A)
        {
            return A + scalar;
        }

        // Vector addition.
        public static Vector operator +(Vector A, Vector B)
        {
            if (A.Shape() != B.Shape())
                throw new Exception($"Vector Element-wise Addition: Incorrect sizes: {A.Shape()} and {B.Shape()}.");

            var resultVector = new Vector(A.Shape());

            for (int i = 0; i < A.Shape(); i++) resultVector.Values[i] = A.Values[i] + B.Values[i];

            return resultVector;
        }

        // Vector division.
        public static Vector operator /(Vector A, Vector B)
        {
            if (A.Shape() != B.Shape())
                throw new Exception($"Vector Element-wise Division: Incorrect sizes: {A.Shape()} and {B.Shape()}.");

            var resultVector = new Vector(A.Shape());

            for (int i = 0; i < A.Shape(); i++)
            {
                if (B.Values[i] == 0)
                    throw new DivideByZeroException();

                resultVector.Values[i] = A.Values[i] / B.Values[i];
            }

            return resultVector;
        }

        // Divide each element of a vector by a scalar.
        public static Vector operator /(Vector A, float scalar)
        {
            if (scalar == 0)
                throw new DivideByZeroException();

            var resultVector = new Vector(A.Shape());
            for (int i = 0; i < A.Shape(); i++) resultVector.Values[i] = A.Values[i] / scalar;
            return resultVector;
        }

        // Reversing the sign of each element of vector.
        public static Vector operator -(Vector vector)
        {
            var resultVector = new Vector(vector.Values.Length);
            for (int i = 0; i < vector.Values.Length; i++) resultVector.Values[i] = -vector.Values[i];
            return resultVector;
        }

        // Subtract from each element of the vector a scalar.
        public static Vector operator -(Vector A, float scalar)
        {
            var resultVector = new Vector(A.Shape());
            for (int i = 0; i < A.Shape(); i++)
                resultVector.Values[i] = A.Values[i] - scalar;
            return resultVector;
        }

        // Subtract scalar from each element of the vector.
        public static Vector operator -(float scalar, Vector A)
        {
            return -A + scalar;
        }

        // Subtraction of vectors.
        public static Vector operator -(Vector A, Vector B)
        {
            if (A.Shape() != B.Shape())
                throw new Exception($"Vector Element-wise Subtraction: Incorrect sizes: {A.Shape()} and {B.Shape()}.");

            var resultVector = new Vector(A.Shape());

            for (int i = 0; i < A.Shape(); i++) resultVector.Values[i] = A.Values[i] - B.Values[i];

            return resultVector;
        }

        // Returns the maximum element of the vector or its index.
        public float ArgMax(bool value = false)
        {

            float maxValue = float.MinValue;
            float maxInd = 0;

            for (int i = 0; i < Shape(); i++)
            {
                if (Values[i] > maxValue)
                {
                    maxValue = Values[i];
                    maxInd = i;

                }
            }
            if (value)
                return maxValue;
            return maxInd;
        }

        // Sum the values of a vector.
        public float Sum()
        {
            float sumResult = 0;
            foreach (float element in Values) sumResult += element;
            return sumResult;
        }

        // Returns the mean of the vector.
        public float Mean()
        {
            int populationSize = Values.Length;
            float sum = Sum();
            return sum / populationSize;
        }

        // Returns the variance of the vector.
        public float Variance()
        {
            int populationSize = Values.Length;
            float mean = Mean();
            float diffSum = 0;
            foreach (float element in Values) diffSum += (float)Math.Pow(element - mean, 2);
            return diffSum / populationSize;
        }

        // Returns the standard deviation of a vector.
        public float StandardDeviation()
        {
            float variance = Variance();
            return (float)Math.Pow(variance, 0.5);
        }

        // Applies the softmax function to a vector.
        public Vector Softmax()
        {
            var softmaxMatrix = new Vector(Values.Length);
            float eSums = 0;
            foreach (float element in Values) eSums += (float)Math.Exp(element);
            for (int i = 0; i < softmaxMatrix.Values.Length; i++)
            {
                if (eSums == 0)
                    throw new DivideByZeroException();

                softmaxMatrix.Values[i] = (float)Math.Exp(Values[i]) / eSums;
            }
            return softmaxMatrix;
        }

        // Applies the hyperbolic tangent function to a vector.
        public Vector Tanh()
        {

            var tanhVector = new Vector(Values.Length);
            for (int i = 0; i < Values.Length; i++)
            {
                double divide = (Math.Exp(Values[i]) + Math.Exp(-Values[i]));

                if (divide == 0)
                    throw new DivideByZeroException();

                tanhVector.Values[i] = (float)((Math.Exp(Values[i]) - Math.Exp(-Values[i])) / divide);
            }

            return tanhVector;
        }

        // One-hot vector encoding.
        public Matrix OneHot(int depth)
        {
            if (Values.Max() >= depth)
                throw new InvalidOperationException("OneHot: Depth is too small");

            var encoded = new Matrix(new int[] { Values.Length, depth });
            for (int i = 0; i < Values.Length; i++)
                encoded.Values[i, (int)Values[i]] = 1;
            return encoded;
        }

    }

    // Represents a matrix and related operations.
    public class Matrix
    {
        // Matrix values.
        public float[,] Values;

        // Thanks to this constructor, we can explicitly define a matrix with specific values.
        public Matrix(float[,] inputValues)
        {
            int[] valuesShape = new int[2] { inputValues.GetLength(0), inputValues.GetLength(1) };

            Values = new float[valuesShape[0], valuesShape[1]];
            for (int i = 0; i < valuesShape[0]; i++)
                for (int j = 0; j < valuesShape[1]; j++) Values[i, j] = inputValues[i, j];
        }

        // Thanks to this constructor we can create a matrix of a certain size with all zeros.
        public Matrix(int[] shape)
        {
            if (shape.Length != 2)
                throw new Exception("Matrix: Wrong matrix shape. The matrix must have dimensions of length two. For example: { 2, 4 }.");

            Values = new float[shape[0], shape[1]];
        }

        // Returns the shape of the matrix.
        public int[] Shape()
        {
            return new int[2] { Values.GetLength(0), Values.GetLength(1) };
        }

        // Prints the shape of the matrix.
        public void PrintShape()
        {
            int[] matrixShape = Shape();
            Console.WriteLine("[{0}, {1}]", matrixShape[0], matrixShape[1]);
        }

        // Prints matrix.
        public void Print()
        {
            int[] matrixShape = Shape();

            for (int i = 0; i < matrixShape[0]; i++)
            {
                for (int j = 0; j < matrixShape[1]; j++) Console.Write("{0} ", Values[i, j]);
                Console.WriteLine(" ");
            }
        }

        // Matrix to vector.
        public Vector FlatMatrix()
        {
            var flattenMatrix = new Vector(Values.Length);

            int j = 0;
            foreach (float i in Values)
            {
                flattenMatrix.Values[j] = i;
                j++;
            }
            return flattenMatrix;
        }

        // Resize matrix to given shape.
        public Matrix ReshapeMatrix(int[] newShape)
        {
            if (newShape.Length != 2)
                throw new Exception("ReshapeMatrix: Wrong new matrix shape. The matrix must have dimensions of length two. For example: { 2, 4 }.");

            if ((newShape[0] * newShape[1]) != Values.Length)
                throw new Exception("Invalid new shape");

            var reshapedMatrix = new Matrix(new int[] { newShape[0], newShape[1] });

            int i = 0;
            int j = 0;

            foreach (float element in Values)
            {
                reshapedMatrix.Values[i, j] = element;

                j++;
                if (j == newShape[1])
                {
                    j = 0;
                    i++;
                }
            }
            return reshapedMatrix;
        }

        // Transposes the matrix.
        public Matrix Transpose()
        {

            int[] matrixShape = Shape();

            var transposedMatrix = new Matrix(new int[] { matrixShape[1], matrixShape[0] });

            for (int i = 0; i < matrixShape[0]; i++)
            {
                for (int j = 0; j < matrixShape[1]; j++)
                    transposedMatrix.Values[j, i] = Values[i, j];
            }

            return transposedMatrix;
        }

        // Matrix Multiplication.
        public static Matrix operator &(Matrix A, Matrix B)
        {
            int[] shapeA = A.Shape();
            int[] shapeB = B.Shape();

            if (shapeA[1] != shapeB[0])
                throw new Exception($"Matrix Multiplication: Invalid matrices size: Sizes are [{shapeA[0]}, {shapeA[1]}] and [{shapeB[0]}, {shapeB[1]}]");

            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeB[1] });

            for (int i = 0; i < shapeA[0]; i++)
            {
                for (int j = 0; j < shapeB[1]; j++)
                {
                    for (int k = 0; k < shapeA[1]; k++)
                        resultMatrix.Values[i, j] = resultMatrix.Values[i, j] + A.Values[i, k] * B.Values[k, j];
                }
            }

            return resultMatrix;
        }

        // Multiply each element of a matrix by a scalar.
        public static Matrix operator *(Matrix A, float scalar)
        {
            int[] shapeA = A.Shape();
            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++)
                    resultMatrix.Values[i, j] = A.Values[i, j] * scalar;

            return resultMatrix;
        }

        // Multiply each element of a matrix by a scalar.
        public static Matrix operator *(float scalar, Matrix A)
        {
            return A * scalar;
        }

        // Matrix Element-wise Multiplication.
        public static Matrix operator *(Matrix A, Matrix B)
        {
            int[] shapeA = A.Shape();
            int[] shapeB = B.Shape();

            if (shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1])
                throw new Exception("Matrix Element-wise Multiplication: Incorrect sizes. Sizes are [{shapeA[0]}, {shapeA[1]}] and [{shapeB[0]}, {shapeB[1]}]");

            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++) resultMatrix.Values[i, j] = A.Values[i, j] * B.Values[i, j];

            return resultMatrix;
        }

        // Add a scalar to each element of the matrix.
        public static Matrix operator +(Matrix A, float scalar)
        {
            int[] shapeA = A.Shape();
            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++)
                    resultMatrix.Values[i, j] = A.Values[i, j] + scalar;
            return resultMatrix;
        }

        // Add a scalar to each element of the matrix.
        public static Matrix operator +(float scalar, Matrix A)
        {
            return A + scalar;
        }

        // Matrix Element-wise Addition.
        public static Matrix operator +(Matrix A, Matrix B)
        {
            int[] shapeA = A.Shape();
            int[] shapeB = B.Shape();

            if (shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1])
                throw new Exception($"Matrix Element-wise Addition: Incorrect sizes. Sizes are [{shapeA[0]}, {shapeA[1]}] and [{shapeB[0]}, {shapeB[1]}]");

            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++) resultMatrix.Values[i, j] = A.Values[i, j] + B.Values[i, j];

            return resultMatrix;
        }

        // Vector is added to each row of the matrix.
        public static Matrix operator +(Matrix A, Vector vector)
        {
            int[] shapeA = A.Shape();

            if (shapeA[1] != vector.Shape())
                throw new Exception($"Matrix-Vector Addition: Wrong matrix or vector shape. Sizes are [{shapeA[0]}, {shapeA[1]}] and [{vector.Shape()}, ]");

            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++)
                    resultMatrix.Values[i, j] = A.Values[i, j] + vector.Values[j];
            return resultMatrix;
        }

        // The scalar is subtracted from each element of the matrix.
        public static Matrix operator -(Matrix A, float scalar)
        {
            int[] shapeA = A.Shape();
            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++)
                    resultMatrix.Values[i, j] = A.Values[i, j] - scalar;
            return resultMatrix;
        }

        // The scalar is subtracted from each element of the matrix.
        public static Matrix operator -(float scalar, Matrix A)
        {
            return -A + scalar;
        }

        // Reversing the sign of each element of matrix.
        public static Matrix operator -(Matrix A)
        {
            int[] shapeA = A.Shape();
            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });
            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++) resultMatrix.Values[i, j] = -A.Values[i, j];
            return resultMatrix;
        }

        // Subtraction of matrices.
        public static Matrix operator -(Matrix A, Matrix B)
        {
            int[] shapeA = A.Shape();
            int[] shapeB = B.Shape();

            if (shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1])
                throw new Exception($"Matrix Subtraction: Incorrect sizes. Sizes are [{shapeA[0]}, {shapeA[1]}] and [{shapeB[0]}, {shapeB[1]}] ");

            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            for (int i = 0; i < shapeA[0]; i++)
            {
                for (int j = 0; j < shapeA[1]; j++)
                    resultMatrix.Values[i, j] = A.Values[i, j] - B.Values[i, j];
            }
            return resultMatrix;
        }

        // Matrix Element-wise division.
        public static Matrix operator /(Matrix A, Matrix B)
        {
            int[] shapeA = A.Shape();
            int[] shapeB = B.Shape();
            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });

            if (shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1])
                throw new Exception($"Matrix Element-wise division: Incorrect sizes. Sizes are [{shapeA[0]}, {shapeA[1]}] and [{shapeB[0]}, {shapeB[1]}] ");

            for (int i = 0; i < shapeA[0]; i++)
            {
                for (int j = 0; j < shapeA[1]; j++)
                {

                    if (B.Values[i, j] == 0)
                        throw new DivideByZeroException();

                    resultMatrix.Values[i, j] = A.Values[i, j] / B.Values[i, j];
                }
            }
            return resultMatrix;
        }

        // Divide each element of a matrix by a scalar.
        public static Matrix operator /(Matrix A, float scalar)
        {
            if (scalar == 0)
                throw new DivideByZeroException();

            int[] shapeA = A.Shape();
            var resultMatrix = new Matrix(new int[] { shapeA[0], shapeA[1] });
            for (int i = 0; i < shapeA[0]; i++)
                for (int j = 0; j < shapeA[1]; j++) resultMatrix.Values[i, j] = A.Values[i, j] / scalar;
            return resultMatrix;
        }

        // Returns the maximum elements of the matrix or its index along the given axis.  
        public Vector ArgMax(int axis, bool values = false)
        {
            if (axis != 0 && axis != 1)
                throw new Exception($"Matrix ArgMax: The axis argument is either 1 or 0. The input value is {axis}.");

            int[] matrixShape = Shape();

            if (axis == 1)
                (matrixShape[0], matrixShape[1]) = (matrixShape[1], matrixShape[0]);

            var maxValues = new Vector(matrixShape[1]);
            var maxInd = new Vector(matrixShape[1]);

            float? currentMax = null;
            int? currentMaxInd = null;

            for (int axis1 = 0; axis1 < matrixShape[1]; axis1++)
            {
                for (int axis2 = 0; axis2 < matrixShape[0]; axis2++)
                {
                    int firstInd = axis2;
                    int secondInd = axis1;

                    if (axis == 1)
                        (firstInd, secondInd) = (secondInd, firstInd);

                    float currentVal = this.Values[firstInd, secondInd];

                    if (!currentMax.HasValue || currentMax < currentVal)
                    {
                        currentMax = currentVal;
                        currentMaxInd = axis2;
                    }

                }
                if (currentMax.HasValue && currentMaxInd.HasValue)
                {
                    maxValues.Values[axis1] = (float)currentMax;
                    maxInd.Values[axis1] = (float)currentMaxInd;
                }
                currentMax = null;
                currentMaxInd = null;
            }
            if (values)
                return maxValues;
            return maxInd;
        }

        // Sum the values of a matrix.
        public float Sum()
        {
            float sumResult = 0;
            foreach (float element in Values) sumResult += element;
            return sumResult;
        }

        // Returns the mean of the matrix.
        public float Mean()
        {
            int populationSize = Values.GetLength(0) * Values.GetLength(1);
            float sum = Sum();
            return sum / populationSize;
        }

        // Returns the variance of the matrix. 
        public float Variance()
        {
            int populationSize = Values.GetLength(0) * Values.GetLength(1);
            float mean = Mean();
            float diffSum = 0;
            foreach (float element in Values) diffSum += (float)Math.Pow(element - mean, 2);
            return diffSum / populationSize;
        }

        // Returns the standard deviation of a matrix.
        public float StandardDeviation()
        {
            float variance = Variance();
            return (float)Math.Pow(variance, 0.5);
        }

        // Reducing the dimension of a matrix to vector by finding the means along a given axis.
        public Vector ReduceMean(int axis)
        {
            if (axis != 0 && axis != 1)
                throw new Exception($"Matrix ReduceMean: The axis argument is either 1 or 0. The input value is {axis}.");

            int[] matrixShape = Shape();

            if (axis == 1)
                (matrixShape[0], matrixShape[1]) = (matrixShape[1], matrixShape[0]);

            var meanValues = new Vector(matrixShape[1]);


            for (int axis1 = 0; axis1 < matrixShape[1]; axis1++)
            {
                float sum = 0;
                for (int axis2 = 0; axis2 < matrixShape[0]; axis2++)
                {
                    int firstInd = axis2;
                    int secondInd = axis1;

                    if (axis == 1)
                        (firstInd, secondInd) = (secondInd, firstInd);

                    sum += Values[firstInd, secondInd];
                }
                meanValues.Values[axis1] = sum / matrixShape[0];
            }
            return meanValues;
        }

        // Applies the softmax function to a matrix.
        public Matrix Softmax()
        {
            var softmaxMatrix = new Matrix(new int[] { Values.GetLength(0), Values.GetLength(1) });

            for (int rows = 0; rows < Values.GetLength(0); rows++)
            {
                float eSums = 0;
                for (int cols = 0; cols < Values.GetLength(1); cols++) eSums += (float)Math.Exp(Values[rows, cols]);

                if (eSums == 0)
                    throw new DivideByZeroException();

                for (int cols = 0; cols < Values.GetLength(1); cols++) softmaxMatrix.Values[rows, cols] = (float)Math.Exp(Values[rows, cols]) / eSums;
            }

            return softmaxMatrix;
        }

        // Applies the ReLU function to a matrix. 
        public Matrix ReLU()
        {
            int[] matrixShape = Shape();
            var reluMatrix = new Matrix(new int[] { matrixShape[0], matrixShape[1] });

            for (int i = 0; i < matrixShape[0]; i++)
            {
                for (int j = 0; j < matrixShape[1]; j++)
                {
                    reluMatrix.Values[i, j] = (float)(Math.Max(0, Values[i, j]));
                }
            }
            return reluMatrix = new Matrix(new int[] { matrixShape[0], matrixShape[1] });
            ;
        }

        // Applies the hyperbolic tangent function to a matrix. 
        public Matrix Tanh()
        {
            int[] matrixShape = Shape();
            var tanhMatrix = new Matrix(new int[] { matrixShape[0], matrixShape[1] });

            for (int i = 0; i < matrixShape[0]; i++)
            {
                for (int j = 0; j < matrixShape[1]; j++)
                {
                    double divide = (Math.Exp(Values[i, j]) + Math.Exp(-Values[i, j]));

                    if (divide == 0)
                        throw new DivideByZeroException();

                    tanhMatrix.Values[i, j] = (float)((Math.Exp(Values[i, j]) - Math.Exp(-Values[i, j])) / divide);
                }
            }
            return tanhMatrix;
        }

        // Applies the activation function according to the name of the function.
        public Matrix Activation(string activation)
        {
            switch (activation)
            {
                case "relu":
                    return ReLU();
                    break;

                case "tanh":
                    return Tanh();
                    break;
                default:
                    return null;
                    break;
            }
        }

        // A special case of the Einstein sum,
        // that is, for matrices A B, einsum(A, B)aij = Aai * Baj.
        public static Tensor EinSum(Matrix A, Matrix B)
        {
            // ai, aj -> aij
            int[] shapeA = A.Shape();
            int[] shapeB = B.Shape();

            var newMatrix = new Tensor(new int[] { shapeA[0], shapeA[1], shapeB[1] });

            for (int a = 0; a < shapeA[0]; a++)
            {
                newMatrix.Values[a] = new float[shapeA[1], shapeB[1]];
                for (int i = 0; i < shapeA[1]; i++)
                {
                    for (int j = 0; j < shapeB[1]; j++)
                    {
                        newMatrix.Values[a][i, j] = A.Values[a, i] * B.Values[a, j];
                    }
                }
            }
            return newMatrix;
        }
    }

    // Represents a tensor and related operations.
    public class Tensor
    {
        // Tensor values.
        public float[][,] Values;

        // Thanks to this constructor, we can explicitly define a tensor with specific values.
        public Tensor(float[][,] inputValues)
        {
            int[] valuesShape = new int[3] { inputValues.Length, inputValues[0].GetLength(0), inputValues[0].GetLength(1) };

            Values = new float[valuesShape[0]][,];

            for (int i = 0; i < valuesShape[0]; i++)
                Values[i] = inputValues[i];
        }

        // Thanks to this constructor we can create a tensor of a certain size with all zeros.
        public Tensor(int[] shape)
        {
            if (shape.Length != 3)
                throw new Exception("Tensor: Wrong tensor shape. The tensor must have dimensions of length three. For example: { 2, 4, 6 }.");

            Values = new float[shape[0]][,];
            for (int i = 0; i < shape[0]; i++)
                Values[i] = new float[shape[1], shape[2]];
        }

        // Thanks to this constructor we can create a tensor of a certain size of first dim with all zeros.
        public Tensor(int length)
        {
            if (length < 1)
                throw new Exception("Tensor: Wrong tensor size. Tensor size cannot be less than 1.");

            Values = new float[length][,];
        }

        // Returns Tensor shape.
        public int[] Shape()
        {
            // Format: { channels, rows, columns }
            return new int[3] { Values.Length, Values[0].GetLength(0), Values[0].GetLength(1) };
        }

        // Prints Tensor shape.
        public void PrintShape()
        {
            int[] valuesShape = Shape();
            Console.WriteLine("[{0}, {1}, {2}]", valuesShape[0], valuesShape[1], valuesShape[2]);
        }

        // Prints Tensor.
        public void Print()
        {
            int[] tensorShape = Shape();

            for (int a = 0; a < tensorShape[0]; a++)
            {
                for (int i = 0; i < tensorShape[1]; i++)
                {
                    for (int j = 0; j < tensorShape[2]; j++)
                        Console.Write("{0} ", Values[a][i, j]);
                    Console.WriteLine(" ");
                }
                Console.WriteLine(" ");
            }
        }

        // Converting Tensor to Matrix.
        public Matrix Tensor2Matrix(int firstAxis)
        {
            int[] tensorShape = Shape();
            int secondAxis = (tensorShape[1] * tensorShape[2]);

            if ((secondAxis * firstAxis) != (tensorShape[0] * tensorShape[1] * tensorShape[2]))
                throw new Exception("Invalid new shape");


            var newTensor = new Matrix(new int[] { firstAxis, secondAxis });

            int i = 0;
            int j = 0;

            foreach (float[,] tensorElement in Values)
            {
                foreach (float element in tensorElement)
                {
                    newTensor.Values[i, j] = element;

                    j++;
                    if (j == secondAxis)
                    {
                        j = 0;
                        i++;
                    }
                }
            }
            return newTensor;
        }

        // Reducing the dimension of a tensor to matrix by finding the means along a given axis.
        public Matrix ReduceMean(int axis)
        {
            if (axis != 0 && axis != 1 && axis != 2)
                throw new Exception($"Tensor ReduceMean: The axis argument is 2 or 1 or 0. The input value is {axis}.");

            int[] matrixShape = Shape();

            // default: axis = 2
            switch (axis)
            {
                case 0:
                    (matrixShape[0], matrixShape[1], matrixShape[2]) = (matrixShape[1], matrixShape[2], matrixShape[0]);
                    break;
                case 1:
                    (matrixShape[1], matrixShape[2]) = (matrixShape[2], matrixShape[1]);
                    break;
            }

            var meanValues = new Matrix(new int[] { matrixShape[0], matrixShape[1] });

            for (int axis1 = 0; axis1 < matrixShape[0]; axis1++)
            {
                for (int axis2 = 0; axis2 < matrixShape[1]; axis2++)
                {
                    float sum = 0;
                    for (int axis3 = 0; axis3 < matrixShape[2]; axis3++)
                    {
                        int firstInd = axis1;
                        int secondInd = axis2;
                        int thirdInd = axis3;

                        switch (axis)
                        {
                            case 0:
                                (firstInd, secondInd, thirdInd) = (thirdInd, firstInd, secondInd);
                                break;
                            case 1:
                                (secondInd, thirdInd) = (thirdInd, secondInd);
                                break;
                        }
                        sum += Values[firstInd][secondInd, thirdInd];
                    }
                    meanValues.Values[axis1, axis2] = sum / matrixShape[2];
                }
            }
            return meanValues;
        }
    }
}