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
            Stopwatch t1 = new Stopwatch();
            t1.Start();
            for (int i = 0; i < 100000; i++)
            {
                net.RunNet(new double[] { 1, 0.5, 0.25 });
            }

            t1.Stop();
            Console.WriteLine(t1.ElapsedMilliseconds);
        }
    }
}
