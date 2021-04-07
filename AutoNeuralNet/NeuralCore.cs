using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Newtonsoft.Json;
using System.Threading.Tasks;

namespace AutoNeuralNet
{
    class NeuralCore
    {
        double[][,] links;

        public void SaveLinks()
        {
            StreamWriter stream = new StreamWriter(@"C:\Users\Mikhail\source\repos\ANNGit\AutoNeuralNet\AutoNeuralNet\links.txt");
            string tlinks = JsonConvert.SerializeObject(links, Formatting.Indented);
            stream.WriteLine(tlinks);
            stream.Close();
            stream.Dispose();
        }

        private double F(double x) => (2 / (1 + Math.Exp(-x))) - 1;

        private double Df(double x) => 0.5 * (1 + F(x)) * (1 - F(x));

        private void GetRandomMatrix(ref double[,] matrix)
        {
            Random random = new Random();
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] = random.NextDouble();
                }
            }
        }

        private void SaveToLog(StreamWriter writer, object[] values)
        {
            foreach (var item in values)
            {
                writer.WriteLine(item.ToString());
            }
        }

        private double[] Dot(double[,] matrix, double[] values)
        {
            double[] final = new double[matrix.GetLength(1)];
            for (int i = 0; i < values.Length; i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    final[j] += values[i] * matrix[i, j];
                }
            }
            return final.Select(i => 2 / (1 + Math.Exp(-i)) - 1).ToArray();
        }

        private double[] TrainigDot(double[,] matrix, double[] values, out double[] nonActivatedOut)
        {
            double[] final = new double[matrix.GetLength(1)];
            for (int i = 0; i < values.Length; i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    final[j] += values[i] * matrix[i, j];
                }
            }
            nonActivatedOut = final;
            return final.Select(i => (2 / (1 + Math.Exp(-i))) - 1).ToArray();
        }

        private double[] TrainRun(double[] values, out double[][] localInputs, out double[][] localOuts)
        {
            localInputs = new double[links.Length][];
            localOuts = new double[links.Length + 1][];
            double[] tmpResults = values;
            localOuts[0] = values;
            for (int i = 0; i < links.Length; i++)
            {
                tmpResults = TrainigDot(links[i], tmpResults, out localInputs[i]);
                localOuts[i + 1] = tmpResults;
            }
            return tmpResults;
        }

        public NeuralCore(int[] config)
        {
            links = new double[config.Length - 1][,];
            for (int i = 0; i < config.Length - 1; i++)
            {
                links[i] = new double[config[i], config[i + 1]];
                GetRandomMatrix(ref links[i]);
            }
        }

        public NeuralCore(string linksPath)
        {
            links = JsonConvert.DeserializeObject<double[][,]>(new StreamReader(linksPath).ReadToEnd());
        }

        public void SetMatrix(double[,] matrix, int layer) { this.links[layer] = matrix; }

        public void SetAllMatrixes(double[][,] m) { links = m; }

        public double[] RunNet(double[] values)
        {
            if (values.Length != links[0].GetLength(0))
            {
                throw new ArgumentOutOfRangeException($"Your inputs length is [{values.Length}], but first matrix rank is [{links[0].GetLength(0)}:{links[0].GetLength(1)}] ");

            }
            else
            {
                double[] tmpResults = values;
                for (int i = 0; i < links.Length; i++)
                {
                    tmpResults = Dot(links[i], tmpResults);
                }
                return tmpResults;
            }

        }
          
        

        public void StartTraining(int iterations, double[][] tests, double[][] correctOutputs, double lmd = 0.001)
        {
            
            for (int N = 0; N < iterations; N++)
            {
                var deltaToSave = 0.0;
                var k = N % tests.Length;
                double[] currentIn = tests[k];
                double[] trueOut = correctOutputs[k];
                double[] currentOut = TrainRun(currentIn, out double[][] localInputs, out double[][] localOuts);
                double[] gradients = new double[localOuts.Last().Length];
                var e = 0.0;
                for (int i = links.Length - 1; i >= 0; i--)
                {

                    if (i == links.Length - 1)
                    {
                        for (int j = 0; j < gradients.Length; j++)
                        {
                            double error = currentOut[j] - trueOut[j];
                            gradients[j] = error * Df(currentOut[j]);
                            e = error;
                        }

                        deltaToSave = gradients[0];
                    }
                    else
                    {
                        double[] nextGradients = new double[localOuts[i + 1].Length];
                        for (int j = 0; j < nextGradients.Length; j++)
                        {
                            for (int a = 0; a < links[i + 1].GetLength(1); a++)
                            {

                                nextGradients[j] += gradients[a];
                                nextGradients[j] *= links[i + 1][j, a];
                                nextGradients[j] *= Df(localOuts[i + 1][j]);
                            }

                        }
                        gradients = nextGradients;
                    }
                    for (int x = 0; x < links[i].GetLength(0) - 1; x++)
                    {
                        for (int y = 0; y < links[i].GetLength(1); y++)
                        {
                            var d = lmd * gradients[y] * localOuts[i][x];
                            //Console.WriteLine($" Delta {d}, X {x}, Y {y}, Layer {i}");
                            links[i][x, y] -= d;
                        }
                    }

                }
            }

        }
    }
}
