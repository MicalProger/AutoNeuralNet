using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoNeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var net = new NeuralCore(new int[] { 3, 2, 1 });
            //net.SetAllMatrixes(new double[][,] { new double[,] { { -0.2, 0.1 }, { 0.3, -0.3 }, { -0.4, -0.4 } }, new double[,] { { 0.2 }, { 0.3 } } });
            double[][] inputs = new double[][] { new double[] {-1, -1, -1}, new double[] { -1, -1, 1 }, new double[] { -1, 1, -1 }, new double[] { -1, 1, 1 },
                                                 new double[] {1, -1, -1}, new double[] {1, -1, 1}, new double[] {1, 1, -1}, new double[] {1, 1, 1}, };
            double[][] outs = new double[][] { new double[] { -1 }, new double[] { 1 }, new double[] { -1 }, new double[] { 1 },
                                               new double[] { -1 }, new double[] { 1 }, new double[] { -1 }, new double[] { -1 }, };

            net.StartTraining(100000, inputs, outs, 0.01);
            for (int i = 0; i < inputs.Length; i++)
            {
                Console.WriteLine($"{net.RunNet(inputs[i])[0]} => {outs[i][0]}");
            }
        }
    }
}
