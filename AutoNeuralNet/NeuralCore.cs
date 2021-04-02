using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoNeuralNet
{
    class NeuralCore
    {
        private void PrintMatrix(int[] m)
        {
            Console.WriteLine($"[{string.Join(", ", m.Select(i => i.ToString()))}]");
        }
        void Prm(double[,] m)
        {
            foreach (var item in m)
            {
                Console.WriteLine(item);
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
        double[][,] links;
        public NeuralCore(int[] config)
        {
            links = new double[config.Length - 1][,];
            for (int i = 0; i < config.Length - 1; i++)
            {
                links[i] = new double[config[i], config[i + 1]];
                GetRandomMatrix(ref links[i]);
            }
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
        private double Df(double x) => 0.5 * (1 + x) * (1 - x);
        public void StartTraining(int iterations, double[][] tests, double[][] correctOutputs, double lmd = 0.01)
        {
            for (int N = 0; N < iterations; N++)
            {
                var k = N % tests.Length;
                double[] currentIn = tests[k];
                double[] trueOut = correctOutputs[k];
                double[] currentOut = RunNet(currentIn);
                for (int j = 0; j < currentOut.Length; j++)
                {
                    var e = currentOut[j] - trueOut[j];
                    var delta = e * Df(currentOut[j]);
                    for (int i = 0; i < links[links.Length - 1].GetLength(0); i++)
                    {
                        links[links.Length - 1][i, j] -= lmd * delta * currentOut[j];
                    }                                     
                }
            }
        }
    }
}
