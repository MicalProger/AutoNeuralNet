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
            net.SetAllMatrixes(new double[][,] { new double[,] { { -0.2, 0.1 }, { 0.3, -0.3 }, { -0.4, -0.4 } }, new double[,] { { 0.2 }, { 0.3 } } });
            net.RunNet(new double[] { 1, 0.5, 0.25 });

            net.StartTraining(10, new double[1][] { new double[] { -1.0, -1.0, -1.0 }   }, new double[][]{new double[] {-1 } });
        }
    }
}
